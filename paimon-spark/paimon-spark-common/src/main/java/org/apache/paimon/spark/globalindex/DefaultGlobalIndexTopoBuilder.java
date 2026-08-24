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
import org.apache.paimon.utils.Pair;
import org.apache.paimon.utils.Range;
import org.apache.paimon.vector.index.VectorGlobalModelTrainer;
import org.apache.paimon.vector.index.VectorGlobalModelTrainers;
import org.apache.paimon.vector.index.VectorTrainingModel;

import org.apache.spark.Partitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2Relation;

import scala.Tuple2;

import javax.annotation.Nullable;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

import static org.apache.paimon.CoreOptions.GLOBAL_INDEX_BUILD_MAX_PARALLELISM;
import static org.apache.paimon.CoreOptions.GLOBAL_INDEX_COLUMN_UPDATE_ACTION;
import static org.apache.paimon.CoreOptions.GLOBAL_INDEX_ROW_COUNT_PER_SHARD;
import static org.apache.paimon.CoreOptions.GlobalIndexColumnUpdateAction.IGNORE;
import static org.apache.paimon.utils.Preconditions.checkArgument;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_ASSIGN_LAZY_STREAMING_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_BACKEND_NATIVE;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_BACKEND_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_TRAIN_MODE_LOCAL;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_TRAIN_MODE_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.GLOBAL_INDEX_FILE_EXTENSION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.IVF_PQ_INDEX_TYPE;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.IVF_PQ_SHARD_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.TRAIN_SAMPLE_ROWS_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.VECTOR_ROUTING_MODEL_FILE_PREFIX;

/** Default topology builder. */
public class DefaultGlobalIndexTopoBuilder implements GlobalIndexTopologyBuilder {

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
        checkArgument(
                CENTROID_TRAIN_MODE_LOCAL.equals(
                        options.getString(
                                CENTROID_TRAIN_MODE_OPTION, CENTROID_TRAIN_MODE_LOCAL)),
                "The centroid-sharded implementation of '%s=%s' currently supports only '%s=%s'.",
                IVF_PQ_SHARD_OPTION,
                IvfPqShard.CENTROID_BASED.optionValue(),
                CENTROID_TRAIN_MODE_OPTION,
                CENTROID_TRAIN_MODE_LOCAL);

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

        Map<String, String> nativeOptions =
                CentroidShardedIvfPqIndexBuilder.nativeOptions(indexType, indexField, options);
        String centroidBackend = centroidBackend(options);
        long trainingSampleRows = trainingSampleRows(indexType, indexField, options);
        List<IndexedSplit> trainingSplits =
                limitSplitsToTrainingSampleRows(splits, trainingSampleRows);
        VectorTrainingModel trainingModel;
        try (VectorGlobalModelTrainer trainer =
                VectorGlobalModelTrainers.create(centroidBackend, indexType, nativeOptions)) {
            collectTrainingSamplesForCentroidBuild(
                    table, readType, indexField, trainingSplits, trainer);
            trainingModel = trainer.finishTraining();
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
        return Collections.singletonList(
                new CommitMessageImpl(
                        BinaryRow.EMPTY_ROW,
                        0,
                        null,
                        dataIncrement,
                        CompactIncrement.emptyIncrement()));
    }

    private static void collectTrainingSamplesForCentroidBuild(
            FileStoreTable table,
            RowType readType,
            DataField indexField,
            List<IndexedSplit> splits,
            VectorGlobalModelTrainer trainer)
            throws IOException {
        for (IndexedSplit split : splits) {
            CentroidShardedIvfPqIndexBuilder indexBuilder =
                    CentroidShardedIvfPqIndexBuilder.forTrainingSamples(
                            table, readType, indexField, split.rowRanges());
            ReadBuilder builder = table.newReadBuilder();
            builder.withReadType(readType);
            try (RecordReader<InternalRow> recordReader = builder.newRead().createReader(split);
                    CloseableIterator<InternalRow> data = recordReader.toCloseableIterator()) {
                for (CentroidShardedIvfPqIndexBuilder.TrainingSample sample :
                        indexBuilder.collectTrainingSamples(data)) {
                    trainer.write(sample.vector(), sample.absoluteRowId());
                }
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
                    "Invalid value for '"
                            + key
                            + "': "
                            + value
                            + ". Must be a positive long.",
                    e);
        }
    }

    static List<IndexedSplit> limitSplitsToTrainingSampleRows(
            List<IndexedSplit> splits, long trainingSampleRows) {
        if (trainingSampleRows < 0) {
            return splits;
        }

        List<IndexedSplit> result = new ArrayList<>();
        long selectedRows = 0;
        for (IndexedSplit split : splits) {
            List<DataFileMeta> selectedFiles = new ArrayList<>();
            for (DataFileMeta file : split.dataSplit().dataFiles()) {
                selectedFiles.add(file);
                selectedRows = saturatedAdd(selectedRows, file.rowCount());
                if (selectedRows > trainingSampleRows) {
                    break;
                }
            }
            if (!selectedFiles.isEmpty()) {
                result.add(copySplitWithDataFiles(split, selectedFiles));
            }
            if (selectedRows > trainingSampleRows) {
                break;
            }
        }
        return result;
    }

    private static IndexedSplit copySplitWithDataFiles(
            IndexedSplit split, List<DataFileMeta> dataFiles) {
        DataSplit dataSplit = split.dataSplit();
        DataSplit sampledDataSplit =
                DataSplit.builder()
                        .withPartition(dataSplit.partition())
                        .withBucket(dataSplit.bucket())
                        .withTotalBuckets(dataSplit.totalBuckets())
                        .withDataFiles(dataFiles)
                        .withBucketPath(dataSplit.bucketPath())
                        .rawConvertible(dataSplit.rawConvertible())
                        .build();
        return new IndexedSplit(sampledDataSplit, split.rowRanges(), split.scores());
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
        List<Pair<byte[], byte[]>> taskList = new ArrayList<>();
        for (IndexedSplit split : splits) {
            CentroidShardedIvfPqIndexBuilder assignBuilder =
                    CentroidShardedIvfPqIndexBuilder.forCentroidAssignment(
                            table,
                            readType,
                            indexField,
                            split.rowRanges(),
                            centroidBackend,
                            indexType,
                            nativeOptions,
                            trainingModelPayload);
            taskList.add(
                    Pair.of(
                            InstantiationUtil.serializeObject(assignBuilder),
                            InstantiationUtil.serializeObject(split)));
        }
        if (taskList.isEmpty()) {
            return Collections.emptyList();
        }

        int scanParallelism = parallelism(taskList.size(), options);
        JavaPairRDD<Integer, CentroidShardedIvfPqIndexBuilder.AssignedVector> assignedVectors =
                centroidAssignLazyStreaming(options)
                        ? javaSparkContext
                                .parallelize(taskList, scanParallelism)
                                .flatMapToPair(
                                        DefaultGlobalIndexTopoBuilder
                                                ::assignCentroidVectorsLazyStreaming)
                        : javaSparkContext
                                .parallelize(taskList, scanParallelism)
                                .flatMapToPair(
                                        DefaultGlobalIndexTopoBuilder::assignCentroidVectors);

        CentroidShardedIvfPqIndexBuilder shardBuilder =
                CentroidShardedIvfPqIndexBuilder.forShardBuild(
                        table, centroidBackend, indexType, nativeOptions, trainingModelPayload);
        byte[] shardBuilderBytes = InstantiationUtil.serializeObject(shardBuilder);
        List<byte[]> shardResultBytes =
                assignedVectors
                        .partitionBy(new CentroidPartitioner(nlist))
                        .mapPartitions(
                                partition -> buildCentroidShardPartition(partition, shardBuilderBytes))
                        .collect();

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
        for (CentroidShardedIvfPqIndexBuilder.ShardBuildResult dataShardResult :
                dataShardResults) {
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

    private static Iterator<
                    Tuple2<Integer, CentroidShardedIvfPqIndexBuilder.AssignedVector>>
            assignCentroidVectors(Pair<byte[], byte[]> task) throws Exception {
        ClassLoader classLoader = DefaultGlobalIndexTopoBuilder.class.getClassLoader();
        CentroidShardedIvfPqIndexBuilder indexBuilder =
                InstantiationUtil.deserializeObject(task.getLeft(), classLoader);
        IndexedSplit split = InstantiationUtil.deserializeObject(task.getRight(), classLoader);
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

    private static Iterator<
                    Tuple2<Integer, CentroidShardedIvfPqIndexBuilder.AssignedVector>>
            assignCentroidVectorsLazyStreaming(Pair<byte[], byte[]> task) throws Exception {
        ClassLoader classLoader = DefaultGlobalIndexTopoBuilder.class.getClassLoader();
        CentroidShardedIvfPqIndexBuilder indexBuilder =
                InstantiationUtil.deserializeObject(task.getLeft(), classLoader);
        IndexedSplit split = InstantiationUtil.deserializeObject(task.getRight(), classLoader);
        ReadBuilder builder = indexBuilder.table().newReadBuilder();
        builder.withReadType(indexBuilder.readType());

        RecordReader<InternalRow> recordReader = null;
        CloseableIterator<InternalRow> data = null;
        try {
            recordReader = builder.newRead().createReader(split);
            data = recordReader.toCloseableIterator();
            return new ClosingIterator<>(
                    indexBuilder.assignCentroidVectorsLazy(data), data, recordReader, indexBuilder);
        } catch (Throwable t) {
            closeQuietly(data, t);
            closeQuietly(recordReader, t);
            closeQuietly(indexBuilder, t);
            throw t;
        }
    }

    private static Iterator<byte[]> buildCentroidShardPartition(
            Iterator<Tuple2<Integer, CentroidShardedIvfPqIndexBuilder.AssignedVector>> partition,
            byte[] shardBuilderBytes)
            throws Exception {
        if (!partition.hasNext()) {
            return Collections.emptyIterator();
        }

        ClassLoader classLoader = DefaultGlobalIndexTopoBuilder.class.getClassLoader();
        CentroidShardedIvfPqIndexBuilder indexBuilder =
                InstantiationUtil.deserializeObject(shardBuilderBytes, classLoader);
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

    private static void closeDataIterator(CloseableIterator<?> data) throws IOException {
        try {
            data.close();
        } catch (Exception e) {
            throw new IOException("Failed to close data iterator.", e);
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

    private static boolean centroidAssignLazyStreaming(Options options) {
        String value = options.getString(CENTROID_ASSIGN_LAZY_STREAMING_OPTION, "false").trim();
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
     * <p>The key of {@code assignedVectors} is already the centroid returned by the global
     * training model. Mapping Spark partition to centroid makes the shuffle contract explicit: all
     * vectors assigned to centroid {@code c} go to Spark partition {@code c}. This avoids relying on
     * the implementation detail that Spark's {@code HashPartitioner} maps {@code Integer c} to
     * {@code c % nlist}.
     */
    private static class CentroidPartitioner extends Partitioner {

        private static final long serialVersionUID = 1L;

        private final int nlist;

        private CentroidPartitioner(int nlist) {
            checkArgument(nlist > 0, "Centroid partitioner requires positive nlist.");
            this.nlist = nlist;
        }

        @Override
        public int numPartitions() {
            return nlist;
        }

        @Override
        public int getPartition(Object key) {
            checkArgument(
                    key instanceof Integer,
                    "Centroid partitioner requires Integer key, but was %s.",
                    key == null ? "null" : key.getClass().getName());
            int centroid = (Integer) key;
            checkArgument(
                    centroid >= 0 && centroid < nlist,
                    "Centroid %s is out of range [0, %s).",
                    centroid,
                    nlist);
            return centroid;
        }
    }

    private static class ClosingIterator<T> implements Iterator<T> {

        private final Iterator<T> delegate;
        private final AutoCloseable[] closeables;
        private boolean closed;

        private ClosingIterator(Iterator<T> delegate, AutoCloseable... closeables) {
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
    }
}
