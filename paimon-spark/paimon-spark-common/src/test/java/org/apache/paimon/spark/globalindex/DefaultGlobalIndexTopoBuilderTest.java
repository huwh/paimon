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

import org.apache.paimon.data.BinaryRow;
import org.apache.paimon.data.BinaryRowWriter;
import org.apache.paimon.data.BinaryVector;
import org.apache.paimon.data.GenericRow;
import org.apache.paimon.data.InternalRow;
import org.apache.paimon.globalindex.IndexedSplit;
import org.apache.paimon.io.DataFileMeta;
import org.apache.paimon.io.DataIncrement;
import org.apache.paimon.options.Options;
import org.apache.paimon.spark.globalindex.sorted.SortedIndexTopoBuilder;
import org.apache.paimon.stats.SimpleStats;
import org.apache.paimon.table.BucketMode;
import org.apache.paimon.table.SpecialFields;
import org.apache.paimon.table.sink.CommitMessage;
import org.apache.paimon.table.source.DataSplit;
import org.apache.paimon.types.DataField;
import org.apache.paimon.types.FloatType;
import org.apache.paimon.types.IntType;
import org.apache.paimon.types.RowType;
import org.apache.paimon.types.VectorType;
import org.apache.paimon.utils.CloseableIterator;
import org.apache.paimon.utils.InstantiationUtil;
import org.apache.paimon.utils.Range;
import org.apache.paimon.vector.index.VectorGlobalModelTrainer;
import org.apache.paimon.vector.index.VectorTrainingModel;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.apache.paimon.CoreOptions.GLOBAL_INDEX_BUILD_MAX_PARALLELISM;
import static org.apache.paimon.CoreOptions.GLOBAL_INDEX_ROW_COUNT_PER_SHARD;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_ASSIGN_LAZY_STREAMING_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.CENTROID_TRAIN_MODE_DISTRIBUTED;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DISTRIBUTED_TRAIN_MAX_PARTIAL_BYTES_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.DISTRIBUTED_TRAIN_MAX_PQ_SAMPLE_BYTES_OPTION;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.MAX_DISTRIBUTED_TRAIN_PARTIAL_BYTES;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Tests for {@link DefaultGlobalIndexTopoBuilder}. */
public class DefaultGlobalIndexTopoBuilderTest {

    @Test
    void testBitmapUsesSortedTopologyBuilder() {
        assertThat(GlobalIndexTopologyBuilderUtils.createTopoBuilder("bitmap"))
                .isInstanceOf(SortedIndexTopoBuilder.class);
    }

    @Test
    void testRowsPerShardUsesMergedBuildOptions() {
        Map<String, String> tableOptions = new HashMap<>();
        tableOptions.put(GLOBAL_INDEX_ROW_COUNT_PER_SHARD.key(), "1000");
        Map<String, String> buildOptions = new HashMap<>();
        buildOptions.put(GLOBAL_INDEX_ROW_COUNT_PER_SHARD.key(), "25");

        assertThat(
                        DefaultGlobalIndexTopoBuilder.rowsPerShard(
                                new Options(tableOptions, buildOptions)))
                .isEqualTo(25L);
    }

    @Test
    void testParallelismUsesBuildMaxParallelism() {
        Options options =
                new Options(
                        Collections.singletonMap(GLOBAL_INDEX_BUILD_MAX_PARALLELISM.key(), "2"));

        assertThat(DefaultGlobalIndexTopoBuilder.parallelism(5, options)).isEqualTo(2);
        assertThat(DefaultGlobalIndexTopoBuilder.parallelism(1, options)).isEqualTo(1);
    }

    @Test
    void testRowsPerShardMustBePositive() {
        Options options =
                new Options(Collections.singletonMap(GLOBAL_INDEX_ROW_COUNT_PER_SHARD.key(), "0"));

        assertThatThrownBy(() -> DefaultGlobalIndexTopoBuilder.rowsPerShard(options))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(
                        "Option 'global-index.row-count-per-shard' must be greater than 0.");
    }

    @Test
    void testMaxParallelismMustBePositive() {
        Options options =
                new Options(
                        Collections.singletonMap(GLOBAL_INDEX_BUILD_MAX_PARALLELISM.key(), "0"));

        assertThatThrownBy(() -> DefaultGlobalIndexTopoBuilder.parallelism(5, options))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(
                        "Option 'global-index.build.max-parallelism' must be greater than 0.");
    }

    @Test
    void testCentroidAssignmentStreamingIsSafeByDefault() {
        assertThat(DefaultGlobalIndexTopoBuilder.centroidAssignLazyStreaming(new Options()))
                .isTrue();

        Options disabled =
                new Options(
                        Collections.singletonMap(CENTROID_ASSIGN_LAZY_STREAMING_OPTION, "false"));
        assertThat(DefaultGlobalIndexTopoBuilder.centroidAssignLazyStreaming(disabled)).isFalse();
    }

    @Test
    void testCentroidTrainingModeValidation() {
        assertThat(DefaultGlobalIndexTopoBuilder.centroidTrainingMode(new Options()))
                .isEqualTo("local");
        assertThat(
                        DefaultGlobalIndexTopoBuilder.centroidTrainingMode(
                                new Options(
                                        Collections.singletonMap(
                                                "ivf.pq.train.mode", " DISTRIBUTED "))))
                .isEqualTo(CENTROID_TRAIN_MODE_DISTRIBUTED);
        assertThatThrownBy(
                        () ->
                                DefaultGlobalIndexTopoBuilder.centroidTrainingMode(
                                        new Options(
                                                Collections.singletonMap(
                                                        "ivf.pq.train.mode", "invalid"))))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("local")
                .hasMessageContaining("distributed");
    }

    @Test
    void testDistributedTrainingRequiresFixedL2Configuration() {
        DataField vectorField = new DataField(0, "vec", new VectorType(4, new FloatType()));
        Map<String, String> autoNlistOptions = distributedNativeOptions();
        autoNlistOptions.put("nlist", "auto");

        assertThatThrownBy(
                        () ->
                                DefaultGlobalIndexTopoBuilder.distributedTrainingSettings(
                                        "ivf-pq", vectorField, autoNlistOptions, new Options()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("fixed positive option 'nlist'");

        Map<String, String> invalidMetricOptions = distributedNativeOptions();
        invalidMetricOptions.put("metric", "inner_product");
        assertThatThrownBy(
                        () ->
                                DefaultGlobalIndexTopoBuilder.distributedTrainingSettings(
                                        "ivf-pq", vectorField, invalidMetricOptions, new Options()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("metric=l2");
    }

    @Test
    void testDistributedPqSampleIsClampedByByteLimit() {
        DataField vectorField = new DataField(0, "vec", new VectorType(4, new FloatType()));
        Map<String, String> optionMap = new HashMap<>();
        optionMap.put("ivf-pq.train.sample-rows", "1000");
        optionMap.put(DISTRIBUTED_TRAIN_MAX_PQ_SAMPLE_BYTES_OPTION, "4096");

        DefaultGlobalIndexTopoBuilder.DistributedTrainingSettings settings =
                DefaultGlobalIndexTopoBuilder.distributedTrainingSettings(
                        "ivf-pq", vectorField, distributedNativeOptions(), new Options(optionMap));
        assertThat(settings.coarseSampleRows()).isEqualTo(1000);
        assertThat(settings.pqSampleRows()).isEqualTo(256);

        optionMap.put(DISTRIBUTED_TRAIN_MAX_PQ_SAMPLE_BYTES_OPTION, "4080");
        assertThatThrownBy(
                        () ->
                                DefaultGlobalIndexTopoBuilder.distributedTrainingSettings(
                                        "ivf-pq",
                                        vectorField,
                                        distributedNativeOptions(),
                                        new Options(optionMap)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("max(256, nlist)=256 vectors");
    }

    @Test
    void testDistributedPqSampleIsClampedByNativeTrainingLimit() {
        DataField vectorField = new DataField(0, "vec", new VectorType(4, new FloatType()));
        Options options =
                new Options(Collections.singletonMap("ivf-pq.train.sample-rows", "1000000"));

        DefaultGlobalIndexTopoBuilder.DistributedTrainingSettings settings =
                DefaultGlobalIndexTopoBuilder.distributedTrainingSettings(
                        "ivf-pq", vectorField, distributedNativeOptions(), options);

        assertThat(settings.coarseSampleRows()).isEqualTo(1_000_000L);
        assertThat(settings.pqSampleRows()).isEqualTo(65_536);
    }

    @Test
    void testDistributedPartialLimitCannotExceedNativeCap() {
        DataField vectorField = new DataField(0, "vec", new VectorType(4, new FloatType()));
        Options options =
                new Options(
                        Collections.singletonMap(
                                DISTRIBUTED_TRAIN_MAX_PARTIAL_BYTES_OPTION,
                                Long.toString((long) MAX_DISTRIBUTED_TRAIN_PARTIAL_BYTES + 1)));

        assertThatThrownBy(
                        () ->
                                DefaultGlobalIndexTopoBuilder.distributedTrainingSettings(
                                        "ivf-pq", vectorField, distributedNativeOptions(), options))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("native accumulator limit");
    }

    @Test
    void testCentroidBuildRequiresBucketUnawareTable() {
        CentroidShardedIvfPqIndexBuildPlanner.validateBucketMode(BucketMode.BUCKET_UNAWARE, "T");

        assertThatThrownBy(
                        () ->
                                CentroidShardedIvfPqIndexBuildPlanner.validateBucketMode(
                                        BucketMode.HASH_FIXED, "T"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("bucket-unaware")
                .hasMessageContaining("HASH_FIXED");
    }

    @Test
    void testCentroidSelectedPartitionMustCoverCompleteTableRowIds() {
        CentroidShardedIvfPqIndexBuildPlanner.validateSelectedPartitionCoverage(
                20L, Arrays.asList(new Range(0, 9), new Range(10, 19)));

        assertThatThrownBy(
                        () ->
                                CentroidShardedIvfPqIndexBuildPlanner
                                        .validateSelectedPartitionCoverage(
                                                20L, Collections.singletonList(new Range(0, 9))))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("complete table row-id range");
    }

    @Test
    void testLimitSplitsToTrainingSampleRowsSelectsFilesByFileMetaRowCount() {
        IndexedSplit split =
                indexedSplit(
                        "bucket-0",
                        dataFile("file-0", 12, 0),
                        dataFile("file-1", 12, 12),
                        dataFile("file-2", 12, 24),
                        dataFile("file-3", 12, 36));

        List<IndexedSplit> sampledSplits =
                DefaultGlobalIndexTopoBuilder.limitSplitsToTrainingSampleRows(
                        Collections.singletonList(split), 30);

        assertThat(sampledSplits).hasSize(3);
        assertThat(
                        sampledSplits.stream()
                                .flatMap(sampled -> fileNames(sampled).stream())
                                .collect(Collectors.toList()))
                .hasSize(3)
                .doesNotHaveDuplicates();
    }

    @Test
    void testLimitSplitsStopsWhenRowLimitIsReachedExactly() {
        IndexedSplit split =
                indexedSplit(
                        "bucket-0",
                        dataFile("file-0", 10, 0),
                        dataFile("file-1", 10, 10),
                        dataFile("file-2", 10, 20));

        List<IndexedSplit> sampledSplits =
                DefaultGlobalIndexTopoBuilder.limitSplitsToTrainingSampleRows(
                        Collections.singletonList(split), 20);

        assertThat(sampledSplits).hasSize(2);
    }

    @Test
    void testDistributedTrainingPartitionQuotasUseValidVectorCounts() {
        assertThat(
                        DefaultGlobalIndexTopoBuilder.trainingPartitionQuotas(
                                Arrays.asList(0L, 100L, 300L), 256L))
                .containsExactly(0L, 100L, 156L);
    }

    @Test
    void testExpectedVectorCountUsesAllFilesAndSaturates() {
        IndexedSplit first =
                indexedSplit(
                        "bucket-0",
                        dataFile("file-0", Long.MAX_VALUE - 5, 0),
                        dataFile("file-1", 10, 1));

        assertThat(
                        DefaultGlobalIndexTopoBuilder.expectedVectorCount(
                                Collections.singletonList(first)))
                .isEqualTo(Long.MAX_VALUE);
    }

    @Test
    void testConfigureExpectedVectorCountForAutoNlist() {
        Map<String, String> nativeOptions = new HashMap<>();
        nativeOptions.put("nlist", "auto");

        DefaultGlobalIndexTopoBuilder.configureExpectedVectorCount(nativeOptions, 1234);

        assertThat(nativeOptions).containsEntry("expected-vector-count", "1234");
    }

    @Test
    void testConfigureExpectedVectorCountPreservesExplicitValue() {
        Map<String, String> nativeOptions = new HashMap<>();
        nativeOptions.put("expected-vector-count", "99");

        DefaultGlobalIndexTopoBuilder.configureExpectedVectorCount(nativeOptions, 1234);

        assertThat(nativeOptions).containsEntry("expected-vector-count", "99");
    }

    @Test
    void testCentroidNativeOptionsForwardIvfPqTuning() {
        Map<String, String> optionMap = new HashMap<>();
        optionMap.put("ivf-pq.pq.code-ratio", "0.125");
        optionMap.put("ivf-pq.target-recall", "0.95");
        optionMap.put("ivf-pq.max-bytes-per-vector", "32");
        optionMap.put("ivf-pq.deployment-profile", "remote");
        optionMap.put("ivf-pq.use-opq", "true");
        optionMap.put("fields.vec.use-opq", "false");

        Map<String, String> nativeOptions =
                CentroidShardedIvfPqIndexBuilder.nativeOptions(
                        "ivf-pq",
                        new DataField(0, "vec", new VectorType(8, new FloatType())),
                        new Options(optionMap));

        assertThat(nativeOptions)
                .containsEntry("pq.code-ratio", "0.125")
                .containsEntry("target-recall", "0.95")
                .containsEntry("max-bytes-per-vector", "32")
                .containsEntry("deployment-profile", "remote")
                .containsEntry("use-opq", "false");
    }

    @Test
    void testCentroidPartitionerBoundsPhysicalPartitions() {
        DefaultGlobalIndexTopoBuilder.CentroidPartitioner partitioner =
                new DefaultGlobalIndexTopoBuilder.CentroidPartitioner(1000, 8);

        assertThat(partitioner.numPartitions()).isEqualTo(8);
        assertThat(partitioner.getPartition(0)).isZero();
        assertThat(partitioner.getPartition(8)).isZero();
        assertThat(partitioner.getPartition(999)).isEqualTo(7);
        assertThatThrownBy(() -> partitioner.getPartition(1000))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("out of range");
    }

    @Test
    void testCentroidCommitUsesLogicalPartitionAndSyntheticBucket() {
        BinaryRow partition = partition(17);

        CommitMessage message =
                DefaultGlobalIndexTopoBuilder.centroidCommitMessage(
                        partition, DataIncrement.emptyIncrement());

        assertThat(message.partition()).isEqualTo(partition);
        assertThat(message.bucket()).isZero();
    }

    @Test
    void testClosingIteratorTaskCompletionCloseIsIdempotent() {
        AtomicInteger closeCount = new AtomicInteger();
        DefaultGlobalIndexTopoBuilder.ClosingIterator<Integer> iterator =
                new DefaultGlobalIndexTopoBuilder.ClosingIterator<>(
                        Collections.singletonList(1).iterator(), closeCount::incrementAndGet);

        assertThat(iterator.next()).isEqualTo(1);
        iterator.closeFromTaskCompletion();
        iterator.closeFromTaskCompletion();

        assertThat(closeCount).hasValue(1);
    }

    @Test
    void testTrainingSamplesAreStreamedWithExactLimit() throws Exception {
        DataField vectorField = new DataField(0, "vec", new VectorType(2, new FloatType()));
        RowType readType = RowType.of(vectorField, SpecialFields.ROW_ID);
        CentroidShardedIvfPqIndexBuilder builder =
                CentroidShardedIvfPqIndexBuilder.forTrainingSamples(
                        null, readType, vectorField, Collections.singletonList(new Range(0, 10)));
        CloseableIterator<InternalRow> rows =
                CloseableIterator.adapterForIterator(
                        Arrays.<InternalRow>asList(
                                        GenericRow.of(
                                                BinaryVector.fromPrimitiveArray(new float[] {1, 2}),
                                                0L),
                                        GenericRow.of(
                                                BinaryVector.fromPrimitiveArray(new float[] {3, 4}),
                                                1L),
                                        GenericRow.of(
                                                BinaryVector.fromPrimitiveArray(new float[] {5, 6}),
                                                2L))
                                .iterator());
        CountingTrainer trainer = new CountingTrainer();

        assertThat(builder.writeTrainingSamples(rows, trainer, 2)).isEqualTo(2);
        assertThat(trainer.count).isEqualTo(2);
        assertThat(rows.hasNext()).isTrue();
    }

    @Test
    void testLazyTrainingVectorsHonorTaskQuota() {
        DataField vectorField = new DataField(0, "vec", new VectorType(2, new FloatType()));
        RowType readType = RowType.of(vectorField, SpecialFields.ROW_ID);
        CentroidShardedIvfPqIndexBuilder builder =
                CentroidShardedIvfPqIndexBuilder.forTrainingSamples(
                        null, readType, vectorField, Collections.singletonList(new Range(0, 10)));
        CloseableIterator<InternalRow> rows =
                CloseableIterator.adapterForIterator(
                        Arrays.<InternalRow>asList(
                                        GenericRow.of(
                                                BinaryVector.fromPrimitiveArray(new float[] {1, 2}),
                                                0L),
                                        GenericRow.of(
                                                BinaryVector.fromPrimitiveArray(new float[] {3, 4}),
                                                1L))
                                .iterator());

        Iterator<float[]> vectors = builder.trainingVectorsLazy(rows, 1);
        assertThat(vectors.next()).containsExactly(1.0f, 2.0f);
        assertThat(vectors.hasNext()).isFalse();
        assertThat(rows.hasNext()).isTrue();
    }

    @Test
    void testBroadcastAssignmentBuilderDoesNotSerializeModelPayload() throws Exception {
        DataField vectorField = new DataField(0, "vec", new VectorType(2, new FloatType()));
        CentroidShardedIvfPqIndexBuilder broadcastBuilder =
                CentroidShardedIvfPqIndexBuilder.forCentroidAssignment(
                        null,
                        RowType.of(vectorField, SpecialFields.ROW_ID),
                        vectorField,
                        Collections.singletonList(new Range(0, 10)),
                        "native",
                        "ivf-pq",
                        Collections.emptyMap());
        CentroidShardedIvfPqIndexBuilder embeddedPayloadBuilder =
                CentroidShardedIvfPqIndexBuilder.forCentroidAssignment(
                        null,
                        RowType.of(vectorField, SpecialFields.ROW_ID),
                        vectorField,
                        Collections.singletonList(new Range(0, 10)),
                        "native",
                        "ivf-pq",
                        Collections.emptyMap(),
                        new byte[1024 * 1024]);

        assertThat(InstantiationUtil.serializeObject(broadcastBuilder).length)
                .isLessThan(InstantiationUtil.serializeObject(embeddedPayloadBuilder).length / 100);
    }

    @Test
    void testTrainingSampleRowsUsesFieldOptionFirst() {
        Map<String, String> optionMap = new HashMap<>();
        optionMap.put("ivf-pq.train.sample-rows", "100");
        optionMap.put("fields.vec.train.sample-rows", "30");
        Options options = new Options(optionMap);

        assertThat(
                        DefaultGlobalIndexTopoBuilder.trainingSampleRows(
                                "ivf-pq", new DataField(0, "vec", new IntType()), options))
                .isEqualTo(30L);
    }

    private static IndexedSplit indexedSplit(String bucketPath, DataFileMeta... dataFiles) {
        DataSplit dataSplit =
                DataSplit.builder()
                        .withPartition(BinaryRow.EMPTY_ROW)
                        .withBucket(0)
                        .withTotalBuckets(1)
                        .withDataFiles(Arrays.asList(dataFiles))
                        .withBucketPath(bucketPath)
                        .rawConvertible(false)
                        .build();
        return new IndexedSplit(dataSplit, Collections.singletonList(new Range(0, 100)), null);
    }

    private static DataFileMeta dataFile(String fileName, long rowCount, long firstRowId) {
        return DataFileMeta.forAppend(
                fileName,
                0,
                rowCount,
                SimpleStats.EMPTY_STATS,
                0,
                0,
                0,
                Collections.emptyList(),
                null,
                null,
                null,
                null,
                firstRowId,
                null);
    }

    private static List<String> fileNames(IndexedSplit split) {
        return split.dataSplit().dataFiles().stream()
                .map(DataFileMeta::fileName)
                .collect(Collectors.toList());
    }

    private static Map<String, String> distributedNativeOptions() {
        Map<String, String> options = new HashMap<>();
        options.put("index.type", "ivf_pq");
        options.put("dimension", "4");
        options.put("nlist", "2");
        options.put("metric", "l2");
        options.put("use-opq", "false");
        return options;
    }

    private static BinaryRow partition(int value) {
        BinaryRow partition = new BinaryRow(1);
        BinaryRowWriter writer = new BinaryRowWriter(partition);
        writer.writeInt(0, value);
        writer.complete();
        return partition;
    }

    private static class CountingTrainer implements VectorGlobalModelTrainer {

        private int count;

        @Override
        public void write(Object vector, long absoluteRowId) {
            count++;
        }

        @Override
        public VectorTrainingModel finishTraining() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void close() {}
    }
}
