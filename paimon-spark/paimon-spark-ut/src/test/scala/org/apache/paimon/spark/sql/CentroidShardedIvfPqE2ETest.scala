/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.paimon.spark.sql

import org.apache.paimon.globalindex.{GlobalIndexIOMeta, IndexFileKind}
import org.apache.paimon.globalindex.io.GlobalIndexFileReader
import org.apache.paimon.index.vector.{VectorIndexTrainer, VectorIndexTraining, VectorIndexWriter}
import org.apache.paimon.spark.PaimonSparkTestBase
import org.apache.paimon.spark.read.SparkVectorSearchBuilderImpl
import org.apache.paimon.table.source.{IndexVectorSearchSplit, RawVectorSearchSplit}
import org.apache.paimon.vector.index.{NativeVectorTrainingModels, VectorGlobalIndexFileMeta, VectorGlobalModelTrainers, VectorIndexMeta}

import java.io.ByteArrayOutputStream
import java.util.LinkedHashMap

import scala.collection.JavaConverters._

/** End-to-end tests for centroid-sharded IVF-PQ build, manifest routing and search. */
class CentroidShardedIvfPqE2ETest extends PaimonSparkTestBase {

  private val IndexType = "ivf-pq"
  private val ScaleEnabledProperty = "paimon.centroid.ivfpq.scale.enabled"
  private val ScaleRowsProperty = "paimon.centroid.ivfpq.scale.rows"
  private val ScaleDimensionProperty = "paimon.centroid.ivfpq.scale.dimension"

  override protected def sparkConf =
    super.sparkConf.set(
      "spark.serializer",
      sys.props.getOrElse(
        "paimon.centroid.ivfpq.spark.serializer",
        "org.apache.spark.serializer.KryoSerializer"))

  test("centroid IVF-PQ build manifest prune and query end to end") {
    assumeCentroidNativeAvailable()

    withTable("T") {
      val dimension = 4
      val nlist = 4
      val padding = (0 until 16).map {
        i => TestPoint(1000 + i, clusteredVector(i % nlist, 200 + i / nlist, dimension))
      }
      val points =
        (0 until 256).map(id => TestPoint(id, clusteredVector(id % nlist, id / nlist, dimension)))
      val allPoints = padding ++ points

      createTable(dimension)
      // Two commits make the queried row's absolute row id different from its business id.
      // This catches an accidental second offset of centroid-shard absolute row ids.
      appendPoints(padding)
      appendPoints(points)

      val buildResult = spark
        .sql(createIndexSql(nlist, pqM = 2, trainSampleRows = allPoints.size))
        .collect()
        .head
      assert(buildResult.getBoolean(0))

      val table = loadTable("T")
      val entries = table.store().newIndexFileHandler().scan(IndexType).asScala.toSeq
      val routingEntries = entries.filter(_.indexFile().fileKind() == IndexFileKind.ROUTING_MODEL)
      val dataEntries = entries.filter(_.indexFile().fileKind() == IndexFileKind.DATA)

      assert(entries.nonEmpty)
      assert(entries.forall(_.partition().getString(0).toString == "p0"))
      assert(entries.map(_.partition()).distinct.size == 1)
      assert(routingEntries.size == 1)
      assert(dataEntries.size > 1, s"Expected multiple centroid shards, got ${dataEntries.size}")
      assert(dataEntries.map(_.indexFile().rowCount()).sum == allPoints.size.toLong)
      assert(
        entries.forall(
          entry =>
            table
              .fileIO()
              .exists(
                table.store().pathFactory().globalIndexFileFactory().toPath(entry.indexFile()))))

      val routingFile = routingEntries.head.indexFile()
      assert(routingFile.rowCount() == 0L)
      assert(routingFile.globalIndexMeta().rowRangeStart() == 0L)
      assert(routingFile.globalIndexMeta().rowRangeEnd() == allPoints.size - 1L)
      val routingManifestMeta =
        VectorIndexMeta.deserialize(routingFile.globalIndexMeta().indexMeta())
      assert(routingManifestMeta.nlist() == nlist)
      assert(routingManifestMeta.modelDigest() != null)

      val shardMetas = dataEntries.map {
        entry =>
          val indexFile = entry.indexFile()
          val meta = VectorIndexMeta.deserialize(indexFile.globalIndexMeta().indexMeta())
          assert(meta.isCentroidShard())
          assert(meta.rowIdEncoding() == VectorIndexMeta.RowIdEncoding.ABSOLUTE_ROW_ID)
          assert(indexFile.globalIndexMeta().rowRangeStart() >= 0L)
          assert(indexFile.globalIndexMeta().rowRangeEnd() < allPoints.size.toLong)
          meta
      }
      assert(shardMetas.map(_.centroid()).distinct.size == dataEntries.size)
      assert(shardMetas.forall(_.modelDigest() == routingManifestMeta.modelDigest()))

      val targetId = 3
      val query = points.find(_.id == targetId).get.vector
      val targetRowId = spark
        .sql(s"SELECT _row_id AS row_id FROM T WHERE id = $targetId AND pt = 'p0'")
        .collect()
        .head
        .getLong(0)
      assert(targetRowId >= padding.size)
      assert(targetRowId != targetId.toLong)

      val routedCentroid = assertTrainingModelRoundTrip(table, routingFile, query, nlist, 2)

      val builder = new SparkVectorSearchBuilderImpl(table)
      builder
        .withVectorColumn("embedding")
        .withVector(query)
        .withLimit(10)
        .withOption("ivf.nprobe", "1")
        .withOption("refine_factor", "8")

      val plan = builder.newVectorScan().scan()
      assert(!plan.splits().asScala.exists(_.isInstanceOf[RawVectorSearchSplit]))
      val indexSplits =
        plan.splits().asScala.collect { case split: IndexVectorSearchSplit => split }
      val routedFiles = indexSplits.flatMap(_.vectorIndexFiles().asScala)
      assert(routedFiles.size == 1, s"nprobe=1 should prune to one shard, got $routedFiles")
      val routedShardMeta =
        VectorIndexMeta.deserialize(routedFiles.head.globalIndexMeta().indexMeta())
      assert(routedShardMeta.centroid() == routedCentroid)

      val directResult = builder.newVectorRead().read(plan)
      assert(directResult.results().contains(targetRowId))

      val queryRows = spark
        .sql(s"""
                |SELECT id, _row_id AS row_id, __paimon_search_score
                |FROM vector_search(
                |  'T', 'embedding', ${vectorSql(query)}, 10,
                |  map('ivf.nprobe', '1', 'refine_factor', '8'))
                |WHERE pt = 'p0'
                |""".stripMargin)
        .collect()
      assert(queryRows.length == 10)

      val rowIdById = spark
        .sql("SELECT id, _row_id AS row_id FROM T WHERE pt = 'p0'")
        .collect()
        .map(row => row.getInt(0) -> row.getLong(1))
        .toMap
      queryRows.foreach {
        row =>
          assert(row.getLong(1) == rowIdById(row.getInt(0)))
          assert(!row.isNullAt(2))
      }
      assert(queryRows.exists(_.getLong(1) == targetRowId))
      assert(queryRows.map(_.getLong(1)).distinct.length == queryRows.length)

      val expectedIds = exactTopK(allPoints, query, 10).toSet
      val actualIds = queryRows.map(_.getInt(0)).toSet
      val recall = actualIds.intersect(expectedIds).size.toDouble / expectedIds.size
      assert(
        recall >= 0.8,
        s"Expected recall@10 >= 0.8, got $recall; actual=$actualIds expected=$expectedIds")
    }
  }

  test("distributed coarse training builds a queryable centroid IVF-PQ index") {
    assumeCentroidNativeAvailable()

    withTable("T") {
      val dimension = 4
      val nlist = 2
      val first =
        (0 until 160).map(id => TestPoint(id, clusteredVector(id % nlist, id / nlist, dimension)))
      val second =
        (160 until 320).map(id => TestPoint(id, clusteredVector(id % nlist, id / nlist, dimension)))
      val points = first ++ second

      createTable(dimension)
      appendPoints(first)
      appendPoints(second)
      assert(loadTable("T").store().newScan().plan().files().size() > 1)

      val result = spark
        .sql(
          createIndexSql(nlist, pqM = 2, trainSampleRows = points.size, trainMode = "distributed"))
        .collect()
        .head
      assert(result.getBoolean(0))

      val entries = loadTable("T").store().newIndexFileHandler().scan(IndexType).asScala.toSeq
      val routingEntries = entries.filter(_.indexFile().fileKind() == IndexFileKind.ROUTING_MODEL)
      val dataEntries = entries.filter(_.indexFile().fileKind() == IndexFileKind.DATA)
      assert(routingEntries.size == 1)
      assert(dataEntries.size > 1)
      assert(dataEntries.map(_.indexFile().rowCount()).sum == points.size.toLong)
      val modelDigest =
        VectorIndexMeta
          .deserialize(routingEntries.head.indexFile().globalIndexMeta().indexMeta())
          .modelDigest()
      assert(
        dataEntries.forall(
          entry =>
            VectorIndexMeta
              .deserialize(entry.indexFile().globalIndexMeta().indexMeta())
              .modelDigest() ==
              modelDigest))

      val target = points(17)
      val table = loadTable("T")
      val builder = new SparkVectorSearchBuilderImpl(table)
      builder
        .withVectorColumn("embedding")
        .withVector(target.vector)
        .withLimit(5)
        .withOption("ivf.nprobe", "1")
        .withOption("refine_factor", "8")
      val plan = builder.newVectorScan().scan()
      assert(!plan.splits().asScala.exists(_.isInstanceOf[RawVectorSearchSplit]))
      val routedFiles = plan
        .splits()
        .asScala
        .collect { case split: IndexVectorSearchSplit => split }
        .flatMap(_.vectorIndexFiles().asScala)
      assert(routedFiles.size == 1)
      assert(!builder.newVectorRead().read(plan).results().isEmpty)

      val rows = spark
        .sql(s"""
                |SELECT id FROM vector_search(
                |  'T', 'embedding', ${vectorSql(target.vector)}, 5,
                |  map('ivf.nprobe', '1', 'refine_factor', '8'))
                |WHERE pt = 'p0'
                |""".stripMargin)
        .collect()
      assert(rows.exists(_.getInt(0) == target.id))
    }
  }

  test("100k centroid IVF-PQ build and query scale check") {
    assume(
      java.lang.Boolean.getBoolean(ScaleEnabledProperty),
      s"Set -D$ScaleEnabledProperty=true in extraJavaTestArgs to run the 100k scale test.")
    assumeCentroidNativeAvailable()

    withTable("T") {
      val rowCount = Integer.getInteger(ScaleRowsProperty, 100000).intValue()
      val dimension = Integer.getInteger(ScaleDimensionProperty, 16).intValue()
      val nlist = 64
      val topK = 10
      require(rowCount >= nlist * 256, s"$ScaleRowsProperty must be at least ${nlist * 256}")
      require(
        dimension == 16 || dimension == 32,
        s"$ScaleDimensionProperty must be either 16 or 32")

      createTable(dimension)
      val vectorExpressions = (0 until dimension).map {
        d =>
          val bit = d % 6
          val center =
            s"CASE WHEN pmod(floor(pmod(id, $nlist) / ${1 << bit}), 2) = 1 " +
              "THEN 100.0 ELSE -100.0 END"
          val jitter =
            s"(pmod(floor(id / $nlist) * ${d + 3} + ${d * 17}, 4093) - 2046) * 0.001"
          s"cast(($center + $jitter) as float)"
      }
      spark
        .range(0, rowCount, 1, 8)
        .selectExpr(
          "cast(id as int) AS id",
          s"array(${vectorExpressions.mkString(",")}) AS embedding",
          "'p0' AS pt")
        .write
        .insertInto("T")

      val buildStarted = System.nanoTime()
      val result = spark
        .sql(createIndexSql(nlist, pqM = 4, trainSampleRows = 50000))
        .collect()
        .head
      val buildMillis = (System.nanoTime() - buildStarted) / 1000000L
      assert(result.getBoolean(0))

      val entries = loadTable("T").store().newIndexFileHandler().scan(IndexType).asScala.toSeq
      val routingEntries = entries.filter(_.indexFile().fileKind() == IndexFileKind.ROUTING_MODEL)
      val dataEntries = entries.filter(_.indexFile().fileKind() == IndexFileKind.DATA)
      assert(entries.forall(_.partition().getString(0).toString == "p0"))
      assert(routingEntries.size == 1)
      assert(dataEntries.size == nlist, s"Expected $nlist non-empty centroid shards")
      assert(dataEntries.map(_.indexFile().rowCount()).sum == rowCount.toLong)

      val queryIds = Seq(123, 17777, 54321, rowCount - 1).filter(_ < rowCount)
      val queryStarted = System.nanoTime()
      val recalls = queryIds.map {
        id =>
          val query = scaleVector(id, nlist, dimension)
          val actual = spark
            .sql(s"""
                    |SELECT id FROM vector_search(
                    |  'T', 'embedding', ${vectorSql(query)}, $topK,
                    |  map('ivf.nprobe', '8', 'refine_factor', '16'))
                    |WHERE pt = 'p0'
                    |""".stripMargin)
            .collect()
            .map(_.getInt(0))
            .toSet
          val expected = exactScaleTopK(rowCount, nlist, dimension, query, topK).toSet
          actual.intersect(expected).size.toDouble / topK
      }
      val queryMillis = (System.nanoTime() - queryStarted) / 1000000L
      val averageRecall = recalls.sum / recalls.size
      assert(averageRecall >= 0.9, s"Expected average recall@10 >= 0.9, got $averageRecall")
      println(
        s"[centroid-ivfpq-scale] rows=$rowCount dimension=$dimension nlist=$nlist " +
          s"shards=${dataEntries.size} buildMs=$buildMillis queries=${queryIds.size} " +
          s"queryMs=$queryMillis recallAt10=$averageRecall")
    }
  }

  private def createTable(dimension: Int): Unit = {
    spark.sql(s"""
                 |CREATE TABLE T (id INT, embedding ARRAY<FLOAT>, pt STRING)
                 |TBLPROPERTIES (
                 |  'bucket' = '-1',
                 |  'global-index.row-count-per-shard' = '10000',
                 |  'global-index.build.max-parallelism' = '8',
                 |  'row-tracking.enabled' = 'true',
                 |  'data-evolution.enabled' = 'true',
                 |  'ivf-pq.dimension' = '$dimension',
                 |  'ivf-pq.metric' = 'l2')
                 |PARTITIONED BY (pt)
                 |""".stripMargin)
  }

  private def createIndexSql(
      nlist: Int,
      pqM: Int,
      trainSampleRows: Int,
      trainMode: String = "local"): String = {
    s"""
       |CALL sys.create_global_index(
       |  table => 'test.T',
       |  partitions => 'pt=p0',
       |  index_column => 'embedding',
       |  index_type => '$IndexType',
       |  options => 'ivf.pq.shard=centroid-based,
       |              ivf.pq.train.mode=$trainMode,
       |              ivf.pq.centroid.backend=native,
       |              ivf-pq.nlist=$nlist,
       |              ivf-pq.pq.m=$pqM,
       |              ivf-pq.use-opq=false,
       |              ivf-pq.train.sample-rows=$trainSampleRows,
       |              global-index.build.max-parallelism=8')
       |""".stripMargin.replaceAll("\\s+", " ")
  }

  private def appendPoints(points: Seq[TestPoint]): Unit = {
    val sparkSession = spark
    import sparkSession.implicits._
    points
      .map(point => (point.id, point.vector.toIndexedSeq, "p0"))
      .toDF("id", "embedding", "pt")
      .write
      .insertInto("T")
  }

  private def assertTrainingModelRoundTrip(
      table: org.apache.paimon.table.FileStoreTable,
      routingFile: org.apache.paimon.index.IndexFileMeta,
      query: Array[Float],
      nlist: Int,
      pqM: Int): Int = {
    val path = table.store().pathFactory().globalIndexFileFactory().toPath(routingFile)
    val ioMeta = new GlobalIndexIOMeta(
      path,
      routingFile.fileSize(),
      routingFile.globalIndexMeta().indexMeta(),
      routingFile.fileKind())
    val fileReader: GlobalIndexFileReader =
      meta => table.fileIO().newInputStream(meta.filePath())
    val artifactHeader = {
      val in = fileReader.getInputStream(ioMeta)
      try VectorGlobalIndexFileMeta.readArtifactHeader(in, routingFile.fileSize()).metadata()
      finally in.close()
    }
    assert(artifactHeader.nlist() == nlist)
    assert(artifactHeader.modelDigest() != null)
    assert(
      artifactHeader.modelDigest() ==
        VectorIndexMeta.deserialize(routingFile.globalIndexMeta().indexMeta()).modelDigest())

    val persistedModel = NativeVectorTrainingModels.load(fileReader, ioMeta)
    val payload = new ByteArrayOutputStream
    val persistedRoute =
      try {
        val route = persistedModel.centroids().nearestCentroids(query, 1)
        persistedModel.serializeNativeModelPayloadTo(payload)
        route
      } finally {
        persistedModel.close()
      }
    assert(persistedRoute.length == 1)
    assert(payload.size() > 0)

    val nativeOptions = new LinkedHashMap[String, String]()
    nativeOptions.put("index.type", "ivf_pq")
    nativeOptions.put("dimension", query.length.toString)
    nativeOptions.put("metric", "l2")
    nativeOptions.put("nlist", nlist.toString)
    nativeOptions.put("pq.m", pqM.toString)
    nativeOptions.put("use-opq", "false")

    val restoredModel =
      VectorGlobalModelTrainers.load("native", IndexType, nativeOptions, payload.toByteArray)
    try {
      assert(restoredModel.centroids().nlist() == nlist)
      assert(restoredModel.centroids().nearestCentroids(query, 1).sameElements(persistedRoute))
      assert(restoredModel.centroids().assignCentroid(query) == persistedRoute.head)
    } finally {
      restoredModel.close()
    }
    persistedRoute.head
  }

  private def assumeCentroidNativeAvailable(): Unit = {
    val result = centroidNativeAvailability()
    result.foreach {
      error =>
        if (System.getProperty("paimon.vindex.native.path") != null) {
          fail(s"Configured centroid IVF-PQ native library is unusable: $error")
        }
        assume(
          condition = false,
          "Centroid IVF-PQ E2E requires a paimon-vector-index Java/native build with " +
            "training-model serde, centroid routing, centroid shard writing and " +
            s"forced-centroid search APIs. $error"
        )
    }
  }

  private def centroidNativeAvailability(): Option[String] = {
    try {
      val options = new LinkedHashMap[String, String]()
      options.put("index.type", "ivf_pq")
      options.put("dimension", "2")
      options.put("metric", "l2")
      options.put("nlist", "1")
      options.put("pq.m", "1")
      options.put("use-opq", "false")
      val training = VectorIndexTrainer.train(options, probeTrainingData(), 512)
      val artifact = training.serialize()
      val restored = VectorIndexTraining.deserialize(artifact)
      val centroid = restored.assignCentroid(Array(0.0f, 0.0f))
      val writer = new VectorIndexWriter(restored, centroid)
      try {
        writer.addIvfPqCentroidVectors(Array(0L), Array(0.0f, 0.0f), 1)
      } finally {
        writer.close()
        restored.close()
        training.close()
      }
      None
    } catch {
      case error: Throwable => Some(error.toString)
    }
  }

  private def probeTrainingData(): Array[Float] = {
    Array.tabulate(1024)(i => ((i * 37) % 257).toFloat * 0.01f)
  }

  private def clusteredVector(cluster: Int, ordinal: Int, dimension: Int): Array[Float] = {
    Array.tabulate(dimension) {
      d =>
        val center = if (((cluster >> (d % 2)) & 1) == 1) 100.0f else -100.0f
        val jitter = (((ordinal * (d + 3) + d * 17) % 127) - 63) * 0.01f
        center + jitter
    }
  }

  private def scaleVector(id: Int, nlist: Int, dimension: Int): Array[Float] = {
    val cluster = id % nlist
    val ordinal = id / nlist
    Array.tabulate(dimension) {
      d =>
        val center = if (((cluster >> (d % 6)) & 1) == 1) 100.0f else -100.0f
        val jitter =
          (Math.floorMod(ordinal.toLong * (d + 3) + d * 17L, 4093L) - 2046L) * 0.001f
        center + jitter
    }
  }

  private def exactTopK(points: Seq[TestPoint], query: Array[Float], limit: Int): Seq[Int] = {
    points.sortBy(point => (squaredL2(point.vector, query), point.id)).take(limit).map(_.id)
  }

  private def exactScaleTopK(
      rowCount: Int,
      nlist: Int,
      dimension: Int,
      query: Array[Float],
      limit: Int): Seq[Int] = {
    val ordering = Ordering.by[(Double, Int), (Double, Int)](identity)
    val nearest = scala.collection.mutable.PriorityQueue.empty[(Double, Int)](ordering)
    var id = 0
    while (id < rowCount) {
      val candidate = (squaredL2(scaleVector(id, nlist, dimension), query), id)
      if (nearest.size < limit) {
        nearest.enqueue(candidate)
      } else if (ordering.lt(candidate, nearest.head)) {
        nearest.dequeue()
        nearest.enqueue(candidate)
      }
      id += 1
    }
    nearest.dequeueAll.reverse.map(_._2)
  }

  private def squaredL2(left: Array[Float], right: Array[Float]): Double = {
    var result = 0.0d
    var i = 0
    while (i < left.length) {
      val delta = left(i).toDouble - right(i).toDouble
      result += delta * delta
      i += 1
    }
    result
  }

  private def vectorSql(vector: Array[Float]): String = {
    // The vector_search parser accepts literal floats inside CreateArray. A CAST expression is
    // semantically equivalent SQL, but is deliberately rejected by that narrow parser contract.
    vector.map(value => s"${java.lang.Float.toString(value)}f").mkString("array(", ",", ")")
  }

  private case class TestPoint(id: Int, vector: Array[Float])
}
