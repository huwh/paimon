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

package org.apache.paimon.spark.globalindex;

import org.apache.paimon.Snapshot;
import org.apache.paimon.data.BinaryRow;
import org.apache.paimon.data.InternalRow;
import org.apache.paimon.fs.Path;
import org.apache.paimon.globalindex.DataEvolutionGlobalIndexRefreshPlanner;
import org.apache.paimon.globalindex.GlobalIndexBuilderUtils;
import org.apache.paimon.globalindex.IndexedSplit;
import org.apache.paimon.globalindex.IvfPqShard;
import org.apache.paimon.globalindex.ResultEntry;
import org.apache.paimon.index.DataEvolutionIndexSourceMeta;
import org.apache.paimon.index.IndexFileMeta;
import org.apache.paimon.io.CompactIncrement;
import org.apache.paimon.io.DataFileMeta;
import org.apache.paimon.io.DataIncrement;
import org.apache.paimon.manifest.IndexManifestEntry;
import org.apache.paimon.manifest.ManifestEntry;
import org.apache.paimon.options.Options;
import org.apache.paimon.partition.PartitionPredicate;
import org.apache.paimon.reader.RecordReader;
import org.apache.paimon.stats.SimpleStatsEvolutions;
import org.apache.paimon.table.FileStoreTable;
import org.apache.paimon.table.sink.CommitMessage;
import org.apache.paimon.table.sink.CommitMessageImpl;
import org.apache.paimon.table.sink.CommitMessageSerializer;
import org.apache.paimon.table.source.DataSplit;
import org.apache.paimon.table.source.ReadBuilder;
import org.apache.paimon.types.DataField;
import org.apache.paimon.types.RowType;
import org.apache.paimon.utils.CloseableIterator;
import org.apache.paimon.utils.InstantiationUtil;
import org.apache.paimon.utils.MurmurHashUtils;
import org.apache.paimon.utils.Pair;
import org.apache.paimon.utils.Range;
import org.apache.paimon.vector.index.NativeDistributedVectorTraining;
import org.apache.paimon.vector.index.VectorGlobalModelTrainer;
import org.apache.paimon.vector.index.VectorGlobalModelTrainers;
import org.apache.paimon.vector.index.VectorTrainingModel;

import org.apache.spark.Partitioner;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2Relation;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.TaskCompletionListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

import scala.Tuple2;

import static org.apache.paimon.CoreOptions.GLOBAL_INDEX_BUILD_MAX_PARALLELISM;
import static org.apache.paimon.CoreOptions.GLOBAL_INDEX_COLUMN_UPDATE_ACTION;
import static org.apache.paimon.CoreOptions.GLOBAL_INDEX_ROW_COUNT_PER_SHARD;
import static org.apache.paimon.CoreOptions.GlobalIndexColumnUpdateAction.IGNORE;
import static org.apache.paimon.utils.Preconditions.checkArgument;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_ASSIGN_LAZY_STREAMING_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_BACKEND_NATIVE;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_BACKEND_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_TRAIN_MODE_DISTRIBUTED;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_TRAIN_MODE_LOCAL;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_TRAIN_MODE_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DEFAULT_DISTRIBUTED_TRAIN_MAX_PQ_SAMPLE_BYTES;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DEFAULT_DISTRIBUTED_TRAIN_MIN_PQ_SAMPLE_ROWS;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DEFAULT_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS_PER_CENTROID;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DISTRIBUTED_TRAIN_MAX_BOOTSTRAP_BYTES_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DISTRIBUTED_TRAIN_MAX_DRIVER_PARTIAL_BYTES_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DISTRIBUTED_TRAIN_MAX_ITERATIONS_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DISTRIBUTED_TRAIN_MAX_PARTIAL_BYTES_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DISTRIBUTED_TRAIN_MAX_PQ_SAMPLE_BYTES_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DISTRIBUTED_TRAIN_MAX_STATE_BYTES_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DISTRIBUTED_TRAIN_PARTITION_BATCH_SIZE_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DISTRIBUTED_TRAIN_TOLERANCE_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.GLOBAL_INDEX_FILE_EXTENSION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.IVF_PQ_INDEX_TYPE;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.IVF_PQ_SHARD_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.MAX_DISTRIBUTED_TRAIN_PARTIAL_BYTES;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.MAX_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.MIN_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.TRAIN_SAMPLE_ROWS_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.VECTOR_ROUTING_MODEL_FILE_PREFIX;

/** Default topology builder. */
public class DefaultGlobalIndexTopoBuilder implements GlobalIndexTopologyBuilder {

    private static final Logger LOG = LoggerFactory.getLogger(DefaultGlobalIndexTopoBuilder.class);

    @Override
    public List<CommitMessage> buildIndex(
            SparkSession spark,
            DataSourceV2Relation relation,
            PartitionPredicate partitionPredicate,
            FileStoreTable table,
            String indexType,
            RowType readType,
            DataField indexField,
            Options options)
            throws IOException {
        return buildIndex(
                spark,
                relation,
                partitionPredicate,
                table,
                indexType,
                readType,
                indexField,
                Collections.emptyList(),
                options);
    }

    @Override
    public List<CommitMessage> buildIndex(
            SparkSession spark,
            DataSourceV2Relation relation,
            PartitionPredicate partitionPredicate,
            FileStoreTable table,
            String indexType,
            RowType readType,
            DataField indexField,
            List<DataField> extraFields,
            Options options)
            throws IOException {
        IvfPqShard ivfPqShard = IvfPqShard.fromOption(options.get(IVF_PQ_SHARD_OPTION));
        if (ivfPqShard == IvfPqShard.CENTROID_BASED) {
            checkArgument(
                    IVF_PQ_INDEX_TYPE.equals(indexType),
                    "Option '%s=%s' is only supported for index type '%s', but was '%s'.",
                    IVF_PQ_SHARD_OPTION,
                    IvfPqShard.CENTROID_BASED.optionValue(),
                    IVF_PQ_INDEX_TYPE,
                    indexType);
            return buildCentroidShardedIvfPqIndex(
                    spark,
                    partitionPredicate,
                    table,
                    indexType,
                    readType,
                    indexField,
                    extraFields,
                    options);
        }

        long rowsPerShard = rowsPerShard(options);

        Snapshot snapshot = table.snapshotManager().latestSnapshot();
        if (snapshot == null) {
            return Collections.emptyList();
        }
        List<DataField> indexFields = new ArrayList<>();
        indexFields.add(indexField);
        indexFields.addAll(extraFields);
        List<IndexManifestEntry> currentIndexes =
                GlobalIndexBuilderUtils.currentIndexEntries(
                        table, snapshot, indexType, indexFields, partitionPredicate);
        List<Range> rowRangesToBuild =
                new ArrayList<>(
                        GlobalIndexBuilderUtils.unindexedRowRanges(snapshot, currentIndexes));
        byte[] sourceMeta = new DataEvolutionIndexSourceMeta(snapshot.id()).serialize();
        boolean detectDataFileChange =
                new Options(table.options(), options.toMap()).get(GLOBAL_INDEX_COLUMN_UPDATE_ACTION)
                        == IGNORE;
        if (rowRangesToBuild.isEmpty() && !detectDataFileChange) {
            return Collections.emptyList();
        }
        List<ManifestEntry> entries =
                table.store()
                        .newScan()
                        .withSnapshot(snapshot)
                        .withPartitionFilter(partitionPredicate)
                        .plan()
                        .files();
        List<IndexManifestEntry> indexesToRefresh = Collections.emptyList();
        if (detectDataFileChange) {
            indexesToRefresh =
                    DataEvolutionGlobalIndexRefreshPlanner.findIndexesToRefresh(
                            table.schemaManager(), entries, currentIndexes, indexFields);
            for (IndexManifestEntry index : indexesToRefresh) {
                rowRangesToBuild.add(index.indexFile().globalIndexMeta().rowRange());
            }
        }
        rowRangesToBuild = Range.sortAndMergeOverlap(rowRangesToBuild, true);
        if (rowRangesToBuild.isEmpty()) {
            return Collections.emptyList();
        }
        // generate splits for each partition && shard
        List<IndexedSplit> splits =
                GlobalIndexBuilderUtils.createShardIndexedSplits(
                        table, entries, rowsPerShard, rowRangesToBuild);

        JavaSparkContext javaSparkContext = new JavaSparkContext(spark.sparkContext());
        List<Pair<byte[], byte[]>> taskList = new ArrayList<>();
        for (IndexedSplit indexedSplit : splits) {
            checkArgument(
                    indexedSplit.rowRanges().size() == 1,
                    "Each IndexedSplit should contain exactly one row range.");
            DefaultGlobalIndexBuilder builder =
                    new DefaultGlobalIndexBuilder(
                            table,
                            indexedSplit.dataSplit().partition(),
                            readType,
                            indexField,
                            extraFields,
                            indexType,
                            indexedSplit.rowRanges().get(0),
                            options,
                            sourceMeta);
            byte[] builderBytes = InstantiationUtil.serializeObject(builder);
            byte[] splitBytes = InstantiationUtil.serializeObject(indexedSplit);
            taskList.add(Pair.of(builderBytes, splitBytes));
        }

        List<CommitMessage> commitMessages = new ArrayList<>();
        if (!taskList.isEmpty()) {
            int parallelism = parallelism(taskList.size(), options);
            List<byte[]> commitMessageBytes =
                    javaSparkContext
                            .parallelize(taskList, parallelism)
                            .map(DefaultGlobalIndexTopoBuilder::buildIndex)
                            .collect();
            commitMessages.addAll(CommitMessageSerializer.deserializeAll(commitMessageBytes));
        }
        for (IndexManifestEntry index : indexesToRefresh) {
            commitMessages.add(
                    new CommitMessageImpl(
                            index.partition(),
                            index.bucket(),
                            null,
                            DataIncrement.deleteIndexIncrement(
                                    Collections.singletonList(index.indexFile())),
                            CompactIncrement.emptyIncrement()));
        }
        return commitMessages;
    }

    private List<CommitMessage> buildCentroidShardedIvfPqIndex(
            SparkSession spark,
            PartitionPredicate partitionPredicate,
            FileStoreTable table,
            String indexType,
            RowType readType,
            DataField indexField,
            List<DataField> extraFields,
            Options options)
            throws IOException {
        checkArgument(
                extraFields.isEmpty(),
                "Option '%s=%s' currently supports only one vector column.",
                IVF_PQ_SHARD_OPTION,
                IvfPqShard.CENTROID_BASED.optionValue());
        String trainingMode = centroidTrainingMode(options);
        String centroidBackend = centroidBackend(options);
        Map<String, String> nativeOptions =
                CentroidShardedIvfPqIndexBuilder.nativeOptions(indexType, indexField, options);
        DistributedTrainingSettings distributedSettings = null;
        if (CENTROID_TRAIN_MODE_DISTRIBUTED.equals(trainingMode)) {
            distributedSettings =
                    distributedTrainingSettings(indexType, indexField, nativeOptions, options);
        }

        Snapshot snapshot = table.snapshotManager().latestSnapshot();
        if (snapshot == null) {
            return Collections.emptyList();
        }

        CentroidShardedIvfPqIndexBuildPlanner planner =
                CentroidShardedIvfPqIndexBuildPlanner.create(
                        table, snapshot, indexType, indexField, partitionPredicate);
        if (planner.isEmpty()) {
            return Collections.emptyList();
        }

        List<DataField> indexFields = planner.indexFields();
        List<Range> rowRangesToBuild = planner.rowRangesToBuild();
        List<IndexedSplit> splits = planner.splits();

        configureExpectedVectorCount(nativeOptions, expectedVectorCount(splits));
        VectorTrainingModel trainingModel;
        if (CENTROID_TRAIN_MODE_DISTRIBUTED.equals(trainingMode)) {
            trainingModel =
                    trainDistributedCentroidModel(
                            spark,
                            table,
                            readType,
                            indexField,
                            splits,
                            indexType,
                            nativeOptions,
                            options,
                            distributedSettings);
        } else {
            long trainingSampleRows = trainingSampleRows(indexType, indexField, options);
            List<IndexedSplit> trainingSplits =
                    limitSplitsToTrainingSampleRows(splits, trainingSampleRows);
            try (VectorGlobalModelTrainer trainer =
                    VectorGlobalModelTrainers.create(centroidBackend, indexType, nativeOptions)) {
                collectTrainingSamplesForCentroidBuild(
                        table, readType, indexField, trainingSplits, trainer, trainingSampleRows);
                trainingModel = trainer.finishTraining();
            }
        }

        List<ResultEntry> resultEntries;
        try {
            resultEntries =
                    buildCentroidDataShards(
                            spark,
                            table,
                            readType,
                            indexField,
                            splits,
                            trainingModel,
                            centroidBackend,
                            indexType,
                            nativeOptions,
                            options);
            if (resultEntries.isEmpty()) {
                return Collections.emptyList();
            }
        } finally {
            trainingModel.close();
        }

        Range indexRange = CentroidShardedIvfPqIndexBuilder.coveringRange(rowRangesToBuild);
        List<IndexFileMeta> indexFileMetas =
                GlobalIndexBuilderUtils.toIndexFileMetas(
                        table.fileIO(),
                        table.store().pathFactory().globalIndexFileFactory(),
                        table.coreOptions(),
                        indexRange,
                        indexFields,
                        indexType,
                        resultEntries);
        DataIncrement dataIncrement = DataIncrement.indexIncrement(indexFileMetas);
        return Collections.singletonList(centroidCommitMessage(planner.partition(), dataIncrement));
    }

    private static VectorTrainingModel trainDistributedCentroidModel(
            SparkSession spark,
            FileStoreTable table,
            RowType readType,
            DataField indexField,
            List<IndexedSplit> splits,
            String indexType,
            Map<String, String> nativeOptions,
            Options options,
            DistributedTrainingSettings settings)
            throws IOException {
        List<TrainingTask> tasks =
                trainingTasks(table, indexField, splits, settings.coarseSampleRows);
        checkArgument(!tasks.isEmpty(), "Distributed vector training requires input files.");
        List<byte[]> taskList = serializeTasks(tasks);

        JavaSparkContext javaSparkContext = new JavaSparkContext(spark.sparkContext());
        TrainingTaskContext taskContext = new TrainingTaskContext(table, readType, indexField);
        JavaRDD<float[]> allVectors =
                javaSparkContext
                        .parallelize(taskList, parallelism(taskList.size(), options))
                        .flatMap(
                                taskBytes ->
                                        readTrainingVectorsLazy(
                                                deserializeTask(taskBytes, "training task"),
                                                taskContext));
        JavaRDD<float[]> vectors =
                limitTrainingVectors(allVectors, settings.coarseSampleRows)
                        .persist(StorageLevel.DISK_ONLY());
        try {
            float[] initialCentroids =
                    SparkVectorTrainingOrchestrator.deterministicInitialCentroids(
                            vectors,
                            settings.dimension,
                            settings.nlist,
                            settings.maxBootstrapBytes);
            NativeCoarseTrainingEngine engine =
                    new NativeCoarseTrainingEngine(
                            settings.dimension,
                            settings.nlist,
                            settings.maxIterations,
                            settings.tolerance,
                            initialCentroids,
                            settings.maxPartialBytes);
            SparkVectorTrainingOrchestrator.TrainingResult coarse =
                    SparkVectorTrainingOrchestrator.trainDistributed(
                            vectors, engine, settings.orchestratorConfiguration());
            if (!coarse.converged()) {
                LOG.warn(
                        "Distributed IVF-PQ coarse training reached the configured limit of {} "
                                + "iterations without satisfying the convergence tolerance.",
                        coarse.iterations());
            }

            checkArgument(
                    coarse.vectorCount()
                                    >= Math.max(
                                            MIN_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS,
                                            (long) settings.nlist)
                            && coarse.vectorCount() <= settings.coarseSampleRows,
                    "Distributed IVF-PQ training requires between max(256, nlist)=%s and %s "
                            + "sample vectors, but found %s.",
                    Math.max(MIN_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS, (long) settings.nlist),
                    settings.coarseSampleRows,
                    coarse.vectorCount());
            int sampleVectorCount =
                    (int) Math.min(coarse.vectorCount(), (long) settings.pqSampleRows);
            float[] rawSample =
                    collectTrainingSample(vectors, settings.dimension, sampleVectorCount);
            return NativeDistributedVectorTraining.trainIvfPq(
                    indexType, nativeOptions, coarse.model(), rawSample, sampleVectorCount);
        } finally {
            vectors.unpersist(false);
        }
    }

    private static float[] collectTrainingSample(
            JavaRDD<float[]> vectors, int dimension, int expectedVectorCount) {
        checkArgument(
                expectedVectorCount > 0,
                "Distributed IVF-PQ training requires at least one non-null vector.");
        int valueCount;
        try {
            valueCount = Math.multiplyExact(expectedVectorCount, dimension);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException(
                    "Distributed PQ sample exceeds the maximum Java array length.", e);
        }
        float[] flattened = new float[valueCount];
        List<float[]> sample = vectors.take(expectedVectorCount);
        checkArgument(
                sample.size() == expectedVectorCount,
                "Distributed training input changed while collecting the PQ sample: "
                        + "expected %s vectors, but found only %s.",
                expectedVectorCount,
                sample.size());
        for (int i = 0; i < expectedVectorCount; i++) {
            float[] vector = sample.get(i);
            checkArgument(
                    vector != null && vector.length == dimension,
                    "Distributed PQ sample vector %s has dimension %s, expected %s.",
                    i,
                    vector == null ? "null" : vector.length,
                    dimension);
            System.arraycopy(vector, 0, flattened, i * dimension, dimension);
        }
        return flattened;
    }

    private static JavaRDD<float[]> limitTrainingVectors(
            JavaRDD<float[]> vectors, long maxTrainingRows) {
        checkArgument(maxTrainingRows > 0, "Training row limit must be greater than 0.");
        int partitionCount = vectors.getNumPartitions();
        List<Tuple2<Integer, Long>> counts =
                new ArrayList<>(
                        vectors.mapPartitionsWithIndex(
                                        (partitionId, partition) -> {
                                            long count = 0L;
                                            while (partition.hasNext()) {
                                                partition.next();
                                                count = Math.addExact(count, 1L);
                                            }
                                            return Collections.singletonList(
                                                            new Tuple2<>(partitionId, count))
                                                    .iterator();
                                        },
                                        true)
                                .collect());
        counts.sort(Comparator.comparingInt(Tuple2::_1));
        checkArgument(
                counts.size() == partitionCount,
                "Distributed training expected %s partition counts, but found %s.",
                partitionCount,
                counts.size());

        List<Long> partitionCounts = new ArrayList<>(partitionCount);
        for (int partitionId = 0; partitionId < partitionCount; partitionId++) {
            Tuple2<Integer, Long> count = counts.get(partitionId);
            checkArgument(
                    count._1() == partitionId,
                    "Distributed training is missing the count for partition %s.",
                    partitionId);
            partitionCounts.add(count._2());
        }
        long[] quotas = trainingPartitionQuotas(partitionCounts, maxTrainingRows);

        return vectors.mapPartitionsWithIndex(
                (partitionId, partition) -> new BoundedIterator<>(partition, quotas[partitionId]),
                true);
    }

    static long[] trainingPartitionQuotas(List<Long> partitionCounts, long maxTrainingRows) {
        checkArgument(maxTrainingRows > 0, "Training row limit must be greater than 0.");
        long[] quotas = new long[partitionCounts.size()];
        long remaining = maxTrainingRows;
        for (int partitionId = 0; partitionId < partitionCounts.size(); partitionId++) {
            long count = partitionCounts.get(partitionId);
            checkArgument(count >= 0, "Training partition count must not be negative.");
            quotas[partitionId] = Math.min(remaining, count);
            remaining -= quotas[partitionId];
        }
        return quotas;
    }

    static CommitMessage centroidCommitMessage(BinaryRow partition, DataIncrement dataIncrement) {
        // A centroid-sharded index covers the complete logical partition rather than one data
        // bucket. Bucket 0 is therefore the synthetic home for this partition-level index.
        return new CommitMessageImpl(
                partition, 0, null, dataIncrement, CompactIncrement.emptyIncrement());
    }

    private static void collectTrainingSamplesForCentroidBuild(
            FileStoreTable table,
            RowType readType,
            DataField indexField,
            List<IndexedSplit> splits,
            VectorGlobalModelTrainer trainer,
            long trainingSampleRows)
            throws IOException {
        long remainingSamples = trainingSampleRows < 0 ? Long.MAX_VALUE : trainingSampleRows;
        for (IndexedSplit split : splits) {
            if (remainingSamples == 0) {
                break;
            }
            CentroidShardedIvfPqIndexBuilder indexBuilder =
                    CentroidShardedIvfPqIndexBuilder.forTrainingSamples(
                            table, readType, indexField, split.rowRanges());
            ReadBuilder builder = table.newReadBuilder();
            builder.withReadType(readType);
            try (RecordReader<InternalRow> recordReader = builder.newRead().createReader(split);
                    CloseableIterator<InternalRow> data = recordReader.toCloseableIterator()) {
                long written = indexBuilder.writeTrainingSamples(data, trainer, remainingSamples);
                remainingSamples -= written;
            } catch (RuntimeException e) {
                throw e;
            } catch (Exception e) {
                throw new IOException("Failed to collect centroid training samples.", e);
            }
        }
    }

    static long trainingSampleRows(String indexType, DataField indexField, Options options) {
        Map<String, String> optionMap = options.toMap();
        String key = "fields." + indexField.name() + "." + TRAIN_SAMPLE_ROWS_OPTION;
        if (!optionMap.containsKey(key)) {
            key = indexType + "." + TRAIN_SAMPLE_ROWS_OPTION;
        }
        if (!optionMap.containsKey(key)) {
            return -1L;
        }

        String value = optionMap.get(key);
        try {
            long sampleRows = Long.parseLong(value.trim());
            checkArgument(sampleRows > 0, "Option '%s' must be greater than 0.", key);
            return sampleRows;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid value for '" + key + "': " + value + ". Must be a positive long.", e);
        }
    }

    static List<IndexedSplit> limitSplitsToTrainingSampleRows(
            List<IndexedSplit> splits, long trainingSampleRows) {
        if (trainingSampleRows < 0) {
            return splits;
        }

        List<IndexedSplit> result = new ArrayList<>();
        long selectedRows = 0;
        for (TrainingFile candidate : sortedTrainingFiles(splits)) {
            result.add(
                    copySplitWithDataFiles(
                            candidate.split, Collections.singletonList(candidate.file)));
            selectedRows = saturatedAdd(selectedRows, candidate.file.rowCount());
            if (selectedRows >= trainingSampleRows) {
                break;
            }
        }
        return result;
    }

    private static List<TrainingTask> trainingTasks(
            FileStoreTable table,
            DataField indexField,
            List<IndexedSplit> splits,
            long maxTrainingRows) {
        checkArgument(maxTrainingRows > 0, "Training row limit must be greater than 0.");
        int fieldIndex = table.schema().logicalRowType().getFieldIndex(indexField.name());
        SimpleStatsEvolutions evolutions =
                new SimpleStatsEvolutions(
                        schemaId -> table.schemaManager().schema(schemaId).fields(),
                        table.schema().id());
        List<TrainingTask> result = new ArrayList<>();
        long remainingEstimatedRows = maxTrainingRows;
        boolean unknownValidCount = false;
        for (TrainingFile candidate : sortedTrainingFiles(splits)) {
            if (remainingEstimatedRows == 0) {
                break;
            }
            IndexedSplit selected =
                    copySplitWithDataFiles(
                            candidate.split, Collections.singletonList(candidate.file));
            long taskLimit;
            if (unknownValidCount) {
                taskLimit = maxTrainingRows;
            } else {
                Long nullCount = selected.dataSplit().nullCount(fieldIndex, evolutions);
                if (nullCount == null) {
                    unknownValidCount = true;
                    taskLimit = maxTrainingRows;
                } else {
                    long validRows = Math.max(0L, candidate.file.rowCount() - nullCount);
                    taskLimit = Math.min(remainingEstimatedRows, validRows);
                    remainingEstimatedRows -= taskLimit;
                }
            }
            if (taskLimit == 0) {
                continue;
            }
            result.add(new TrainingTask(selected, taskLimit));
        }
        return result;
    }

    private static List<TrainingFile> sortedTrainingFiles(List<IndexedSplit> splits) {
        List<TrainingFile> candidates = new ArrayList<>();
        for (IndexedSplit split : splits) {
            for (DataFileMeta file : split.dataSplit().dataFiles()) {
                candidates.add(new TrainingFile(split, file));
            }
        }
        candidates.sort(
                Comparator.comparingLong(TrainingFile::samplingKey)
                        .thenComparing(candidate -> candidate.file.fileName()));
        return candidates;
    }

    private static Iterator<float[]> readTrainingVectorsLazy(
            TrainingTask task, TrainingTaskContext context) throws Exception {
        IndexedSplit split = task.split;
        CentroidShardedIvfPqIndexBuilder indexBuilder = context.newBuilder(split.rowRanges());
        ReadBuilder readBuilder = indexBuilder.table().newReadBuilder();
        readBuilder.withReadType(indexBuilder.readType());

        RecordReader<InternalRow> recordReader = null;
        CloseableIterator<InternalRow> rows = null;
        try {
            recordReader = readBuilder.newRead().createReader(split);
            rows = recordReader.toCloseableIterator();
            ClosingIterator<float[]> iterator =
                    new ClosingIterator<>(
                            indexBuilder.trainingVectorsLazy(rows, task.maxVectors),
                            rows,
                            recordReader,
                            indexBuilder);
            registerTaskCompletionClose(iterator);
            return iterator;
        } catch (Throwable t) {
            closeQuietly(rows, t);
            closeQuietly(recordReader, t);
            closeQuietly(indexBuilder, t);
            throw t;
        }
    }

    static String centroidTrainingMode(Options options) {
        String mode =
                options.getString(CENTROID_TRAIN_MODE_OPTION, CENTROID_TRAIN_MODE_LOCAL)
                        .trim()
                        .toLowerCase(Locale.ROOT);
        checkArgument(
                CENTROID_TRAIN_MODE_LOCAL.equals(mode)
                        || CENTROID_TRAIN_MODE_DISTRIBUTED.equals(mode),
                "Option '%s' supports only '%s' or '%s', but was '%s'.",
                CENTROID_TRAIN_MODE_OPTION,
                CENTROID_TRAIN_MODE_LOCAL,
                CENTROID_TRAIN_MODE_DISTRIBUTED,
                mode);
        return mode;
    }

    static DistributedTrainingSettings distributedTrainingSettings(
            String indexType,
            DataField indexField,
            Map<String, String> nativeOptions,
            Options options) {
        checkArgument(
                IVF_PQ_INDEX_TYPE.equals(indexType),
                "Distributed vector training supports only index type '%s'.",
                IVF_PQ_INDEX_TYPE);
        int dimension = parsePositiveNativeInt(nativeOptions, "dimension");
        int nlist = parsePositiveNativeInt(nativeOptions, "nlist");
        checkArgument(
                "l2".equals(normalizeNativeOption(nativeOptions.get("metric"))),
                "Distributed vector training requires option 'metric=l2'.");
        String useOpq = nativeOptions.get("use-opq");
        checkArgument(
                useOpq == null || "false".equals(normalizeNativeOption(useOpq)),
                "Distributed vector training requires option 'use-opq=false'.");
        nativeOptions.put("metric", "l2");
        nativeOptions.put("use-opq", "false");

        int maxIterations =
                positiveIntOption(
                        options,
                        DISTRIBUTED_TRAIN_MAX_ITERATIONS_OPTION,
                        SparkVectorTrainingOrchestrator.DEFAULT_MAX_ITERATIONS);
        checkArgument(
                maxIterations <= SparkVectorTrainingOrchestrator.MAX_ITERATIONS_LIMIT,
                "Option '%s' must not exceed %s.",
                DISTRIBUTED_TRAIN_MAX_ITERATIONS_OPTION,
                SparkVectorTrainingOrchestrator.MAX_ITERATIONS_LIMIT);
        double tolerance =
                nonNegativeDoubleOption(options, DISTRIBUTED_TRAIN_TOLERANCE_OPTION, 1.0e-4d);
        int maxStateBytes =
                positiveIntOption(
                        options,
                        DISTRIBUTED_TRAIN_MAX_STATE_BYTES_OPTION,
                        SparkVectorTrainingOrchestrator.DEFAULT_MAX_STATE_BYTES);
        int maxPartialBytes =
                positiveIntOption(
                        options,
                        DISTRIBUTED_TRAIN_MAX_PARTIAL_BYTES_OPTION,
                        SparkVectorTrainingOrchestrator.DEFAULT_MAX_PARTIAL_BYTES);
        checkArgument(
                maxPartialBytes <= MAX_DISTRIBUTED_TRAIN_PARTIAL_BYTES,
                "Option '%s' must not exceed the native accumulator limit of %s bytes.",
                DISTRIBUTED_TRAIN_MAX_PARTIAL_BYTES_OPTION,
                MAX_DISTRIBUTED_TRAIN_PARTIAL_BYTES);
        long maxDriverPartialBytes =
                positiveLongOption(
                        options,
                        DISTRIBUTED_TRAIN_MAX_DRIVER_PARTIAL_BYTES_OPTION,
                        SparkVectorTrainingOrchestrator.DEFAULT_MAX_DRIVER_PARTIAL_BYTES);
        checkArgument(
                maxDriverPartialBytes >= maxPartialBytes,
                "Option '%s' must be at least option '%s'.",
                DISTRIBUTED_TRAIN_MAX_DRIVER_PARTIAL_BYTES_OPTION,
                DISTRIBUTED_TRAIN_MAX_PARTIAL_BYTES_OPTION);
        int partitionBatchSize =
                positiveIntOption(
                        options,
                        DISTRIBUTED_TRAIN_PARTITION_BATCH_SIZE_OPTION,
                        SparkVectorTrainingOrchestrator.DEFAULT_PARTITION_BATCH_SIZE);
        checkArgument(
                partitionBatchSize <= SparkVectorTrainingOrchestrator.MAX_PARTITION_BATCH_SIZE,
                "Option '%s' must not exceed %s.",
                DISTRIBUTED_TRAIN_PARTITION_BATCH_SIZE_OPTION,
                SparkVectorTrainingOrchestrator.MAX_PARTITION_BATCH_SIZE);
        long maxBootstrapBytes =
                positiveLongOption(
                        options,
                        DISTRIBUTED_TRAIN_MAX_BOOTSTRAP_BYTES_OPTION,
                        SparkVectorTrainingOrchestrator.DEFAULT_MAX_BOOTSTRAP_BYTES);

        long requestedSampleRows = trainingSampleRows(indexType, indexField, options);
        if (requestedSampleRows < 0) {
            requestedSampleRows =
                    Math.max(
                            DEFAULT_DISTRIBUTED_TRAIN_MIN_PQ_SAMPLE_ROWS,
                            (long) nlist * DEFAULT_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS_PER_CENTROID);
        }
        checkArgument(
                requestedSampleRows >= MIN_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS,
                "Distributed IVF-PQ option '%s' must be at least %s rows.",
                TRAIN_SAMPLE_ROWS_OPTION,
                MIN_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS);
        checkArgument(
                requestedSampleRows >= nlist,
                "Distributed IVF-PQ option '%s' must be at least nlist=%s rows.",
                TRAIN_SAMPLE_ROWS_OPTION,
                nlist);
        long maxPqSampleBytes =
                positiveLongOption(
                        options,
                        DISTRIBUTED_TRAIN_MAX_PQ_SAMPLE_BYTES_OPTION,
                        DEFAULT_DISTRIBUTED_TRAIN_MAX_PQ_SAMPLE_BYTES);
        long bytesPerVector = Math.multiplyExact((long) dimension, Float.BYTES);
        long rowsAllowedByBytes = maxPqSampleBytes / bytesPerVector;
        long pqSampleRows =
                Math.min(
                        requestedSampleRows,
                        Math.min(
                                MAX_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS,
                                Math.min(
                                        rowsAllowedByBytes, (long) Integer.MAX_VALUE / dimension)));
        checkArgument(
                pqSampleRows >= Math.max(MIN_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS, (long) nlist),
                "Option '%s' allows only %s vectors of dimension %s, but distributed IVF-PQ "
                        + "requires room for at least max(256, nlist)=%s vectors.",
                DISTRIBUTED_TRAIN_MAX_PQ_SAMPLE_BYTES_OPTION,
                pqSampleRows,
                dimension,
                Math.max(MIN_DISTRIBUTED_TRAIN_PQ_SAMPLE_ROWS, (long) nlist));

        return new DistributedTrainingSettings(
                dimension,
                nlist,
                maxIterations,
                tolerance,
                maxStateBytes,
                maxPartialBytes,
                maxDriverPartialBytes,
                partitionBatchSize,
                maxBootstrapBytes,
                requestedSampleRows,
                (int) pqSampleRows);
    }

    private static int parsePositiveNativeInt(Map<String, String> nativeOptions, String key) {
        String value = nativeOptions.get(key);
        checkArgument(
                value != null && !"auto".equalsIgnoreCase(value.trim()),
                "Distributed vector training requires a fixed positive option '%s'.",
                key);
        try {
            int parsed = Integer.parseInt(value.trim());
            checkArgument(
                    parsed > 0,
                    "Distributed vector training requires a fixed positive option '%s'.",
                    key);
            return parsed;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Distributed vector training requires a fixed positive option '"
                            + key
                            + "', but was '"
                            + value
                            + "'.",
                    e);
        }
    }

    private static String normalizeNativeOption(@Nullable String value) {
        return value == null ? null : value.trim().toLowerCase(Locale.ROOT);
    }

    private static int positiveIntOption(Options options, String key, int defaultValue) {
        long value = positiveLongOption(options, key, defaultValue);
        checkArgument(value <= Integer.MAX_VALUE, "Option '%s' exceeds the integer limit.", key);
        return (int) value;
    }

    private static long positiveLongOption(Options options, String key, long defaultValue) {
        String raw = options.getString(key, Long.toString(defaultValue));
        try {
            long value = Long.parseLong(raw.trim());
            checkArgument(value > 0, "Option '%s' must be greater than 0.", key);
            return value;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid value for '" + key + "': " + raw + ". Must be a positive long.", e);
        }
    }

    private static double nonNegativeDoubleOption(
            Options options, String key, double defaultValue) {
        String raw = options.getString(key, Double.toString(defaultValue));
        try {
            double value = Double.parseDouble(raw.trim());
            checkArgument(
                    !Double.isNaN(value) && !Double.isInfinite(value) && value >= 0.0d,
                    "Option '%s' must be finite and non-negative.",
                    key);
            return value;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid value for '" + key + "': " + raw + ". Must be a non-negative double.",
                    e);
        }
    }

    static long expectedVectorCount(List<IndexedSplit> splits) {
        long expectedVectorCount = 0;
        for (IndexedSplit split : splits) {
            for (DataFileMeta file : split.dataSplit().dataFiles()) {
                expectedVectorCount = saturatedAdd(expectedVectorCount, file.rowCount());
            }
        }
        return expectedVectorCount;
    }

    static void configureExpectedVectorCount(
            Map<String, String> nativeOptions, long expectedVectorCount) {
        String nlist = nativeOptions.get("nlist");
        if ((nlist == null || "auto".equalsIgnoreCase(nlist.trim()))
                && !nativeOptions.containsKey("expected-vector-count")) {
            nativeOptions.put("expected-vector-count", Long.toString(expectedVectorCount));
        }
    }

    private static IndexedSplit copySplitWithDataFiles(
            IndexedSplit split, List<DataFileMeta> dataFiles) {
        DataSplit dataSplit = split.dataSplit();
        DataSplit.Builder builder =
                DataSplit.builder()
                        .withSnapshot(dataSplit.snapshotId())
                        .withPartition(dataSplit.partition().copy())
                        .withBucket(dataSplit.bucket())
                        .withTotalBuckets(dataSplit.totalBuckets())
                        .withDataFiles(dataFiles)
                        .withBucketPath(dataSplit.bucketPath())
                        .isStreaming(dataSplit.isStreaming())
                        .rawConvertible(dataSplit.rawConvertible());
        dataSplit.deletionFiles().ifPresent(builder::withDataDeletionFiles);
        DataSplit sampledDataSplit = builder.build();
        return new IndexedSplit(sampledDataSplit, split.rowRanges(), split.scores());
    }

    private static <T extends java.io.Serializable> List<byte[]> serializeTasks(List<T> tasks)
            throws IOException {
        List<byte[]> serialized = new ArrayList<>(tasks.size());
        for (T task : tasks) {
            serialized.add(InstantiationUtil.serializeObject(task));
        }
        return serialized;
    }

    private static <T> T deserializeTask(byte[] bytes, String description) throws IOException {
        try {
            return InstantiationUtil.deserializeObject(
                    bytes, DefaultGlobalIndexTopoBuilder.class.getClassLoader());
        } catch (ClassNotFoundException e) {
            throw new IOException("Failed to deserialize " + description + '.', e);
        }
    }

    private static long saturatedAdd(long left, long right) {
        long result = left + right;
        if (((left ^ result) & (right ^ result)) < 0) {
            return Long.MAX_VALUE;
        }
        return result;
    }

    private static List<ResultEntry> buildCentroidDataShards(
            SparkSession spark,
            FileStoreTable table,
            RowType readType,
            DataField indexField,
            List<IndexedSplit> splits,
            VectorTrainingModel trainingModel,
            String centroidBackend,
            String indexType,
            Map<String, String> nativeOptions,
            Options options)
            throws IOException {
        JavaSparkContext javaSparkContext = new JavaSparkContext(spark.sparkContext());
        byte[] trainingModelPayload = serializeTrainingModelPayload(trainingModel);
        int nlist = trainingModel.centroids().nlist();
        List<byte[]> taskList = serializeTasks(splits);
        if (taskList.isEmpty()) {
            return Collections.emptyList();
        }

        Broadcast<byte[]> trainingModelBroadcast = javaSparkContext.broadcast(trainingModelPayload);
        List<byte[]> shardResultBytes;
        try {
            int scanParallelism = parallelism(taskList.size(), options);
            AssignmentTaskContext assignmentContext =
                    new AssignmentTaskContext(
                            table,
                            readType,
                            indexField,
                            centroidBackend,
                            indexType,
                            nativeOptions,
                            trainingModelBroadcast);
            JavaPairRDD<Integer, CentroidShardedIvfPqIndexBuilder.AssignedVector> assignedVectors =
                    centroidAssignLazyStreaming(options)
                            ? javaSparkContext
                                    .parallelize(taskList, scanParallelism)
                                    .flatMapToPair(
                                            taskBytes ->
                                                    assignCentroidVectorsLazyStreaming(
                                                            deserializeTask(
                                                                    taskBytes, "centroid split"),
                                                            assignmentContext))
                            : javaSparkContext
                                    .parallelize(taskList, scanParallelism)
                                    .flatMapToPair(
                                            taskBytes ->
                                                    assignCentroidVectors(
                                                            deserializeTask(
                                                                    taskBytes, "centroid split"),
                                                            assignmentContext));

            CentroidShardedIvfPqIndexBuilder shardBuilder =
                    CentroidShardedIvfPqIndexBuilder.forShardBuild(
                            table, centroidBackend, indexType, nativeOptions);
            byte[] shardBuilderBytes = InstantiationUtil.serializeObject(shardBuilder);
            int shardParallelism = parallelism(nlist, options);
            shardResultBytes =
                    assignedVectors
                            .repartitionAndSortWithinPartitions(
                                    new CentroidPartitioner(nlist, shardParallelism))
                            .mapPartitions(
                                    partition ->
                                            buildCentroidShardPartition(
                                                    partition,
                                                    shardBuilderBytes,
                                                    trainingModelBroadcast.value()))
                            .collect();
        } finally {
            trainingModelBroadcast.destroy(false);
        }

        List<CentroidShardedIvfPqIndexBuilder.ShardBuildResult> dataShardResults =
                new ArrayList<>();
        ClassLoader classLoader = DefaultGlobalIndexTopoBuilder.class.getClassLoader();
        for (byte[] resultBytes : shardResultBytes) {
            try {
                CentroidShardedIvfPqIndexBuilder.ShardBuildResult result =
                        InstantiationUtil.deserializeObject(resultBytes, classLoader);
                dataShardResults.add(result);
            } catch (ClassNotFoundException e) {
                throw new IOException("Failed to deserialize centroid shard build result.", e);
            }
        }

        if (dataShardResults.isEmpty()) {
            return Collections.emptyList();
        }

        List<ResultEntry> resultEntries = new ArrayList<>();
        for (CentroidShardedIvfPqIndexBuilder.ShardBuildResult dataShardResult : dataShardResults) {
            resultEntries.add(dataShardResult.toResultEntry());
        }

        String routingModelFileName = newRoutingModelFileName();
        CentroidShardedIvfPqIndexBuilder globalIndexFileBuilder =
                CentroidShardedIvfPqIndexBuilder.forGlobalIndexFile(
                        table, trainingModel, nativeOptions);
        resultEntries.add(
                globalIndexFileBuilder.writeVectorGlobalIndexFile(
                        routingModelFileName,
                        indexType,
                        indexField.id(),
                        null,
                        coveringSplitRange(splits)));
        return resultEntries;
    }

    private static String newRoutingModelFileName() {
        return VECTOR_ROUTING_MODEL_FILE_PREFIX
                + '-'
                + UUID.randomUUID()
                + GLOBAL_INDEX_FILE_EXTENSION;
    }

    private static Range coveringSplitRange(List<IndexedSplit> splits) {
        List<Range> ranges = new ArrayList<>();
        for (IndexedSplit split : splits) {
            ranges.addAll(split.rowRanges());
        }
        return CentroidShardedIvfPqIndexBuilder.coveringRange(ranges);
    }

    private static byte[] serializeTrainingModelPayload(VectorTrainingModel trainingModel)
            throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        trainingModel.serializeNativeModelPayloadTo(out);
        return out.toByteArray();
    }

    private static Iterator<Tuple2<Integer, CentroidShardedIvfPqIndexBuilder.AssignedVector>>
            assignCentroidVectors(IndexedSplit split, AssignmentTaskContext context)
                    throws Exception {
        CentroidShardedIvfPqIndexBuilder indexBuilder = context.newBuilder(split.rowRanges());
        ReadBuilder builder = indexBuilder.table().newReadBuilder();
        builder.withReadType(indexBuilder.readType());

        try {
            try (RecordReader<InternalRow> recordReader = builder.newRead().createReader(split);
                    CloseableIterator<InternalRow> data = recordReader.toCloseableIterator()) {
                return indexBuilder.assignCentroidVectors(data).iterator();
            }
        } finally {
            indexBuilder.close();
        }
    }

    private static Iterator<Tuple2<Integer, CentroidShardedIvfPqIndexBuilder.AssignedVector>>
            assignCentroidVectorsLazyStreaming(IndexedSplit split, AssignmentTaskContext context)
                    throws Exception {
        CentroidShardedIvfPqIndexBuilder indexBuilder = context.newBuilder(split.rowRanges());
        ReadBuilder builder = indexBuilder.table().newReadBuilder();
        builder.withReadType(indexBuilder.readType());

        RecordReader<InternalRow> recordReader = null;
        CloseableIterator<InternalRow> data = null;
        try {
            recordReader = builder.newRead().createReader(split);
            data = recordReader.toCloseableIterator();
            ClosingIterator<Tuple2<Integer, CentroidShardedIvfPqIndexBuilder.AssignedVector>>
                    closingIterator =
                            new ClosingIterator<>(
                                    indexBuilder.assignCentroidVectorsLazy(data),
                                    data,
                                    recordReader,
                                    indexBuilder);
            registerTaskCompletionClose(closingIterator);
            return closingIterator;
        } catch (Throwable t) {
            closeQuietly(data, t);
            closeQuietly(recordReader, t);
            closeQuietly(indexBuilder, t);
            throw t;
        }
    }

    private static Iterator<byte[]> buildCentroidShardPartition(
            Iterator<Tuple2<Integer, CentroidShardedIvfPqIndexBuilder.AssignedVector>> partition,
            byte[] shardBuilderBytes,
            byte[] modelPayload)
            throws Exception {
        if (!partition.hasNext()) {
            return Collections.emptyIterator();
        }

        ClassLoader classLoader = DefaultGlobalIndexTopoBuilder.class.getClassLoader();
        CentroidShardedIvfPqIndexBuilder indexBuilder =
                InstantiationUtil.deserializeObject(shardBuilderBytes, classLoader);
        indexBuilder.setBroadcastTrainingModelPayload(modelPayload);
        try {
            List<byte[]> resultEntries = new ArrayList<>();
            for (CentroidShardedIvfPqIndexBuilder.ShardBuildResult result :
                    indexBuilder.buildCentroidShards(partition)) {
                resultEntries.add(InstantiationUtil.serializeObject(result));
            }
            return resultEntries.iterator();
        } finally {
            indexBuilder.close();
        }
    }

    private static void closeQuietly(AutoCloseable closeable, Throwable error) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (Throwable closeError) {
            error.addSuppressed(closeError);
        }
    }

    private static String centroidBackend(Options options) {
        String backend = options.getString(CENTROID_BACKEND_OPTION, CENTROID_BACKEND_NATIVE).trim();
        checkArgument(
                CENTROID_BACKEND_NATIVE.equals(backend),
                "Option '%s' supports only '%s', but was '%s'.",
                CENTROID_BACKEND_OPTION,
                CENTROID_BACKEND_NATIVE,
                backend);
        return backend;
    }

    static boolean centroidAssignLazyStreaming(Options options) {
        String value = options.getString(CENTROID_ASSIGN_LAZY_STREAMING_OPTION, "true").trim();
        checkArgument(
                "true".equalsIgnoreCase(value) || "false".equalsIgnoreCase(value),
                "Option '%s' supports only 'true' or 'false', but was '%s'.",
                CENTROID_ASSIGN_LAZY_STREAMING_OPTION,
                value);
        return Boolean.parseBoolean(value);
    }

    static long rowsPerShard(Options options) {
        long rowsPerShard = options.get(GLOBAL_INDEX_ROW_COUNT_PER_SHARD);
        checkArgument(
                rowsPerShard > 0,
                "Option 'global-index.row-count-per-shard' must be greater than 0.");
        return rowsPerShard;
    }

    static int parallelism(int taskCount, Options options) {
        int maxParallelism = options.get(GLOBAL_INDEX_BUILD_MAX_PARALLELISM);
        checkArgument(
                maxParallelism > 0,
                "Option 'global-index.build.max-parallelism' must be greater than 0.");
        return Math.min(taskCount, maxParallelism);
    }

    private static byte[] buildIndex(Pair<byte[], byte[]> builderAndSplits) throws Exception {
        ClassLoader classLoader = DefaultGlobalIndexBuilder.class.getClassLoader();
        DefaultGlobalIndexBuilder indexBuilder =
                InstantiationUtil.deserializeObject(builderAndSplits.getLeft(), classLoader);
        byte[] dataSplitBytes = builderAndSplits.getRight();
        IndexedSplit split = InstantiationUtil.deserializeObject(dataSplitBytes, classLoader);
        ReadBuilder builder = indexBuilder.table().newReadBuilder();
        builder.withReadType(indexBuilder.readType());

        try (RecordReader<InternalRow> recordReader = builder.newRead().createReader(split);
                CloseableIterator<InternalRow> data = recordReader.toCloseableIterator()) {
            CommitMessage commitMessage = indexBuilder.build(data);
            return new CommitMessageSerializer().serialize(commitMessage);
        }
    }

    /**
     * Groups files into shards by partition. This method delegates to the generic global index
     * build planner and keeps the previous test surface stable.
     *
     * @param entriesByPartition manifest entries grouped by partition
     * @param rowsPerShard number of rows per shard
     * @param pathFactory path factory for creating bucket paths
     * @return map of partition to shard splits
     */
    public static Map<BinaryRow, List<IndexedSplit>> groupFilesIntoShardsByPartition(
            Map<BinaryRow, List<ManifestEntry>> entriesByPartition,
            long rowsPerShard,
            BiFunction<BinaryRow, Integer, Path> pathFactory) {
        return groupFilesIntoShardsByPartition(entriesByPartition, rowsPerShard, pathFactory, null);
    }

    public static Map<BinaryRow, List<IndexedSplit>> groupFilesIntoShardsByPartition(
            Map<BinaryRow, List<ManifestEntry>> entriesByPartition,
            long rowsPerShard,
            BiFunction<BinaryRow, Integer, Path> pathFactory,
            @Nullable List<Range> rowRangesToBuild) {
        List<ManifestEntry> entries =
                entriesByPartition.values().stream()
                        .flatMap(List::stream)
                        .collect(Collectors.toList());
        return GlobalIndexBuilderUtils.createShardIndexedSplits(
                        entries,
                        rowsPerShard,
                        (partition, bucket) -> pathFactory.apply(partition, bucket).toString(),
                        rowRangesToBuild)
                .stream()
                .collect(Collectors.groupingBy(split -> split.dataSplit().partition()));
    }

    /**
     * Partitions assigned vectors directly by centroid.
     *
     * <p>The key is the centroid returned by the global model. Multiple centroids may share one
     * physical Spark partition so large {@code nlist} values do not create an unbounded number of
     * tasks. Modulo mapping is deterministic and keeps all vectors for one centroid together.
     */
    static class CentroidPartitioner extends Partitioner {

        private static final long serialVersionUID = 1L;

        private final int centroidCount;
        private final int physicalPartitionCount;

        CentroidPartitioner(int centroidCount, int physicalPartitionCount) {
            checkArgument(centroidCount > 0, "Centroid partitioner requires positive nlist.");
            checkArgument(
                    physicalPartitionCount > 0 && physicalPartitionCount <= centroidCount,
                    "Centroid physical partition count must be in [1, %s], but was %s.",
                    centroidCount,
                    physicalPartitionCount);
            this.centroidCount = centroidCount;
            this.physicalPartitionCount = physicalPartitionCount;
        }

        @Override
        public int numPartitions() {
            return physicalPartitionCount;
        }

        @Override
        public int getPartition(Object key) {
            checkArgument(
                    key instanceof Integer,
                    "Centroid partitioner requires Integer key, but was %s.",
                    key == null ? "null" : key.getClass().getName());
            int centroid = (Integer) key;
            checkArgument(
                    centroid >= 0 && centroid < centroidCount,
                    "Centroid %s is out of range [0, %s).",
                    centroid,
                    centroidCount);
            return centroid % physicalPartitionCount;
        }
    }

    static class ClosingIterator<T> implements Iterator<T> {

        private final Iterator<T> delegate;
        private final AutoCloseable[] closeables;
        private boolean closed;

        ClosingIterator(Iterator<T> delegate, AutoCloseable... closeables) {
            this.delegate = delegate;
            this.closeables = closeables;
        }

        @Override
        public boolean hasNext() {
            try {
                boolean hasNext = delegate.hasNext();
                if (!hasNext) {
                    close();
                }
                return hasNext;
            } catch (RuntimeException | Error e) {
                closeQuietly(e);
                throw e;
            }
        }

        @Override
        public T next() {
            try {
                return delegate.next();
            } catch (RuntimeException | Error e) {
                closeQuietly(e);
                throw e;
            }
        }

        private void close() {
            if (closed) {
                return;
            }
            closed = true;
            RuntimeException error = null;
            for (AutoCloseable closeable : closeables) {
                if (closeable == null) {
                    continue;
                }
                try {
                    closeable.close();
                } catch (Exception e) {
                    if (error == null) {
                        error =
                                new RuntimeException(
                                        "Failed to close centroid assignment iterator.", e);
                    } else {
                        error.addSuppressed(e);
                    }
                }
            }
            if (error != null) {
                throw error;
            }
        }

        private void closeQuietly(Throwable error) {
            if (closed) {
                return;
            }
            closed = true;
            for (AutoCloseable closeable : closeables) {
                if (closeable == null) {
                    continue;
                }
                try {
                    closeable.close();
                } catch (Throwable closeError) {
                    error.addSuppressed(closeError);
                }
            }
        }

        void closeFromTaskCompletion() {
            if (closed) {
                return;
            }
            closed = true;
            for (AutoCloseable closeable : closeables) {
                if (closeable == null) {
                    continue;
                }
                try {
                    closeable.close();
                } catch (Throwable closeError) {
                    LOG.warn(
                            "Failed to close centroid assignment resource on task completion.",
                            closeError);
                }
            }
        }
    }

    private static class BoundedIterator<T> implements Iterator<T> {

        private final Iterator<T> delegate;
        private long remaining;

        private BoundedIterator(Iterator<T> delegate, long limit) {
            this.delegate = delegate;
            this.remaining = limit;
        }

        @Override
        public boolean hasNext() {
            return remaining > 0 && delegate.hasNext();
        }

        @Override
        public T next() {
            if (!hasNext()) {
                throw new java.util.NoSuchElementException();
            }
            remaining--;
            return delegate.next();
        }
    }

    private static void registerTaskCompletionClose(ClosingIterator<?> iterator) {
        TaskContext taskContext = TaskContext.get();
        if (taskContext != null) {
            taskContext.addTaskCompletionListener(
                    (TaskCompletionListener) ignored -> iterator.closeFromTaskCompletion());
        }
    }

    private static class AssignmentTaskContext implements java.io.Serializable {

        private static final long serialVersionUID = 1L;

        private final FileStoreTable table;
        private final RowType readType;
        private final DataField indexField;
        private final String backend;
        private final String indexType;
        private final Map<String, String> nativeOptions;
        private final Broadcast<byte[]> trainingModelBroadcast;

        private AssignmentTaskContext(
                FileStoreTable table,
                RowType readType,
                DataField indexField,
                String backend,
                String indexType,
                Map<String, String> nativeOptions,
                Broadcast<byte[]> trainingModelBroadcast) {
            this.table = table;
            this.readType = readType;
            this.indexField = indexField;
            this.backend = backend;
            this.indexType = indexType;
            this.nativeOptions = nativeOptions;
            this.trainingModelBroadcast = trainingModelBroadcast;
        }

        private CentroidShardedIvfPqIndexBuilder newBuilder(List<Range> rowRanges) {
            CentroidShardedIvfPqIndexBuilder builder =
                    CentroidShardedIvfPqIndexBuilder.forCentroidAssignment(
                            table,
                            readType,
                            indexField,
                            rowRanges,
                            backend,
                            indexType,
                            nativeOptions);
            builder.setBroadcastTrainingModelPayload(trainingModelBroadcast.value());
            return builder;
        }
    }

    private static class TrainingTaskContext implements java.io.Serializable {

        private static final long serialVersionUID = 1L;

        private final FileStoreTable table;
        private final RowType readType;
        private final DataField indexField;

        private TrainingTaskContext(FileStoreTable table, RowType readType, DataField indexField) {
            this.table = table;
            this.readType = readType;
            this.indexField = indexField;
        }

        private CentroidShardedIvfPqIndexBuilder newBuilder(List<Range> rowRanges) {
            return CentroidShardedIvfPqIndexBuilder.forTrainingSamples(
                    table, readType, indexField, rowRanges);
        }
    }

    static class TrainingTask implements java.io.Serializable {

        private static final long serialVersionUID = 1L;

        private final IndexedSplit split;
        private final long maxVectors;

        private TrainingTask(IndexedSplit split, long maxVectors) {
            this.split = split;
            this.maxVectors = maxVectors;
        }

        IndexedSplit split() {
            return split;
        }

        long maxVectors() {
            return maxVectors;
        }
    }

    static class DistributedTrainingSettings {

        private final int dimension;
        private final int nlist;
        private final int maxIterations;
        private final double tolerance;
        private final int maxStateBytes;
        private final int maxPartialBytes;
        private final long maxDriverPartialBytes;
        private final int partitionBatchSize;
        private final long maxBootstrapBytes;
        private final long coarseSampleRows;
        private final int pqSampleRows;

        private DistributedTrainingSettings(
                int dimension,
                int nlist,
                int maxIterations,
                double tolerance,
                int maxStateBytes,
                int maxPartialBytes,
                long maxDriverPartialBytes,
                int partitionBatchSize,
                long maxBootstrapBytes,
                long coarseSampleRows,
                int pqSampleRows) {
            this.dimension = dimension;
            this.nlist = nlist;
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;
            this.maxStateBytes = maxStateBytes;
            this.maxPartialBytes = maxPartialBytes;
            this.maxDriverPartialBytes = maxDriverPartialBytes;
            this.partitionBatchSize = partitionBatchSize;
            this.maxBootstrapBytes = maxBootstrapBytes;
            this.coarseSampleRows = coarseSampleRows;
            this.pqSampleRows = pqSampleRows;
        }

        private SparkVectorTrainingOrchestrator.Configuration orchestratorConfiguration() {
            return new SparkVectorTrainingOrchestrator.Configuration(
                    maxIterations,
                    maxStateBytes,
                    maxPartialBytes,
                    maxDriverPartialBytes,
                    partitionBatchSize);
        }

        int pqSampleRows() {
            return pqSampleRows;
        }

        long coarseSampleRows() {
            return coarseSampleRows;
        }
    }

    private static class TrainingFile {

        private final IndexedSplit split;
        private final DataFileMeta file;

        private TrainingFile(IndexedSplit split, DataFileMeta file) {
            this.split = split;
            this.file = file;
        }

        private long samplingKey() {
            byte[] key =
                    (file.fileName() + ':' + file.firstRowId()).getBytes(StandardCharsets.UTF_8);
            return Integer.toUnsignedLong(MurmurHashUtils.hashBytes(key));
        }
    }
}
