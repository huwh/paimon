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

import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.AdvanceResult;
import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.Configuration;
import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.PartitionAccumulator;
import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.PartitionPartial;
import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.TrainingEngine;
import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.TrainingResult;

import org.apache.spark.TaskContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Tests for {@link SparkVectorTrainingOrchestrator}. */
public class SparkVectorTrainingOrchestratorTest {

    private static SparkSession spark;
    private static JavaSparkContext sparkContext;

    @BeforeAll
    static void startSpark() {
        scala.Option<SparkSession> existing = SparkSession.getActiveSession();
        if (existing.isEmpty()) {
            existing = SparkSession.getDefaultSession();
        }
        if (!existing.isEmpty()) {
            existing.get().stop();
        }
        SparkSession.clearActiveSession();
        SparkSession.clearDefaultSession();
        spark =
                SparkSession.builder()
                        .appName(SparkVectorTrainingOrchestratorTest.class.getSimpleName())
                        .master("local[2,2]")
                        .config("spark.ui.enabled", "false")
                        .config("spark.driver.bindAddress", "127.0.0.1")
                        .getOrCreate();
        sparkContext = JavaSparkContext.fromSparkContext(spark.sparkContext());
    }

    @AfterAll
    static void stopSpark() {
        if (spark != null) {
            spark.stop();
            SparkSession.clearActiveSession();
            SparkSession.clearDefaultSession();
        }
    }

    @Test
    void testLocalAndDistributedUseTheSameStateMachine() throws Exception {
        List<List<float[]>> partitions =
                Arrays.asList(
                        Arrays.asList(vector(1), vector(2)),
                        Collections.emptyList(),
                        Arrays.asList(vector(3), vector(4)));
        Configuration configuration = new Configuration(5, 128, 1024);

        TrainingResult local =
                SparkVectorTrainingOrchestrator.trainLocal(
                        partitions, new SummingEngine(3), configuration);
        JavaRDD<float[]> distributedInput =
                sparkContext
                        .parallelizePairs(
                                Arrays.asList(
                                        new scala.Tuple2<>(0, vector(1)),
                                        new scala.Tuple2<>(0, vector(2)),
                                        new scala.Tuple2<>(2, vector(3)),
                                        new scala.Tuple2<>(2, vector(4))))
                        .partitionBy(new ExactPartitioner(3))
                        .values();
        TrainingResult distributed =
                SparkVectorTrainingOrchestrator.trainDistributed(
                        distributedInput, new SummingEngine(3), configuration);

        assertThat(distributed.model()).containsExactly(local.model());
        assertThat(distributed.finalState()).containsExactly(local.finalState());
        assertThat(distributed.iterations()).isEqualTo(3);
        assertThat(distributed.converged()).isTrue();
        assertThat(distributed.vectorCount()).isEqualTo(4);
        assertThat(distributed.partitionCount()).isEqualTo(3);
    }

    @Test
    void testPartitionInputIsFedInBoundedBatches() throws Exception {
        BatchRecordingEngine engine = new BatchRecordingEngine();

        TrainingResult result =
                SparkVectorTrainingOrchestrator.trainLocal(
                        Collections.singletonList(
                                Arrays.asList(
                                        vector(1), vector(2), vector(3), vector(4), vector(5))),
                        engine,
                        new Configuration(1, 128, 128, 1024, 2));

        assertThat(engine.batchSizes).containsExactly(2, 2, 1);
        assertThat(result.vectorCount()).isEqualTo(5);
    }

    @Test
    void testEngineCanLowerConfiguredBatchSize() throws Exception {
        BatchRecordingEngine engine = new BatchRecordingEngine(2);

        SparkVectorTrainingOrchestrator.trainLocal(
                Collections.singletonList(
                        Arrays.asList(vector(1), vector(2), vector(3), vector(4), vector(5))),
                engine,
                new Configuration(1, 128, 128, 1024, 5));

        assertThat(engine.batchSizes).containsExactly(2, 2, 1);
    }

    @Test
    void testDistributedTrainingRetriesWithoutDoubleCounting() throws Exception {
        JavaRDD<float[]> input = sparkContext.parallelize(Arrays.asList(vector(1), vector(2)), 2);

        TrainingResult result =
                SparkVectorTrainingOrchestrator.trainDistributed(
                        input, new SummingEngine(1, 0), new Configuration(1, 128, 1024));

        assertThat(result.vectorCount()).isEqualTo(2);
        assertThat(decodeLong(result.model())).isEqualTo(3);
    }

    @Test
    void testEmptyPartitionsAreNotMerged() throws Exception {
        JavaRDD<float[]> input = sparkContext.parallelize(Collections.singletonList(vector(7)), 4);

        TrainingResult result =
                SparkVectorTrainingOrchestrator.trainDistributed(
                        input, new SummingEngine(1), new Configuration(1, 128, 1024));

        assertThat(result.vectorCount()).isEqualTo(1);
        assertThat(result.partitionCount()).isEqualTo(4);
        assertThat(decodeLong(result.model())).isEqualTo(7);
    }

    @Test
    void testAllEmptyPartitionsAreRejected() {
        JavaRDD<float[]> input = sparkContext.parallelize(Collections.emptyList(), 3);

        assertThatThrownBy(
                        () ->
                                SparkVectorTrainingOrchestrator.trainDistributed(
                                        input,
                                        new SummingEngine(1),
                                        new Configuration(1, 128, 1024)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("every partition was empty");
    }

    @Test
    void testCanonicalizePartialsSortsAndDeduplicatesRetry() {
        List<PartitionPartial> result =
                SparkVectorTrainingOrchestrator.canonicalizePartials(
                        Arrays.asList(partial(1, 2), partial(0, 1), partial(1, 2)), 2, 16, 32);

        assertThat(result).extracting(PartitionPartial::partitionId).containsExactly(0, 1);
        assertThat(result).extracting(PartitionPartial::vectorCount).containsExactly(1L, 1L);
    }

    @Test
    void testCanonicalizePartialsRejectsConflictingRetry() {
        assertThatThrownBy(
                        () ->
                                SparkVectorTrainingOrchestrator.canonicalizePartials(
                                        Arrays.asList(partial(0, 1), partial(0, 2)), 1, 16, 32))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("Conflicting training partials");
    }

    @Test
    void testCanonicalizePartialsRejectsMissingPartition() {
        assertThatThrownBy(
                        () ->
                                SparkVectorTrainingOrchestrator.canonicalizePartials(
                                        Collections.singletonList(partial(1, 2)), 2, 16, 32))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("Missing training partial for logical partition 0");
    }

    @Test
    void testPayloadLimitsAreEnforced() {
        assertThatThrownBy(
                        () ->
                                SparkVectorTrainingOrchestrator.canonicalizePartials(
                                        Collections.singletonList(
                                                new PartitionPartial(0, 1, new byte[9])),
                                        1,
                                        8,
                                        64))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("exceeding the 8 byte limit");

        assertThatThrownBy(
                        () ->
                                SparkVectorTrainingOrchestrator.canonicalizePartials(
                                        Arrays.asList(
                                                new PartitionPartial(0, 1, new byte[8]),
                                                new PartitionPartial(1, 1, new byte[8])),
                                        2,
                                        8,
                                        12))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("require 16 driver bytes");

        assertThatThrownBy(
                        () ->
                                SparkVectorTrainingOrchestrator.deterministicInitialCentroids(
                                        Collections.singletonList(vector(1)), 1, 1, 3))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("bootstrap requires 4 bytes");
    }

    @Test
    void testIterationLimitStopsAnUnconvergedEngine() throws Exception {
        TrainingResult result =
                SparkVectorTrainingOrchestrator.trainLocal(
                        Collections.singletonList(Collections.singletonList(vector(2))),
                        new SummingEngine(10),
                        new Configuration(2, 128, 1024));

        assertThat(result.iterations()).isEqualTo(2);
        assertThat(result.converged()).isFalse();
    }

    @Test
    void testConfigurationRejectsUnboundedIterationCount() {
        assertThatThrownBy(
                        () ->
                                new Configuration(
                                        SparkVectorTrainingOrchestrator.MAX_ITERATIONS_LIMIT + 1,
                                        128,
                                        1024))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("max iterations");
    }

    @Test
    void testAccumulatorClosesOnFailure() {
        CloseTrackingEngine engine = new CloseTrackingEngine();

        assertThatThrownBy(
                        () ->
                                SparkVectorTrainingOrchestrator.trainLocal(
                                        Collections.singletonList(
                                                Collections.singletonList(vector(1))),
                                        engine,
                                        new Configuration(1, 128, 1024)))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("injected failure");
        assertThat(engine.closed).isTrue();
    }

    private static float[] vector(float value) {
        return new float[] {value};
    }

    private static PartitionPartial partial(int partitionId, long value) {
        return new PartitionPartial(partitionId, 1L, encodeLong(value));
    }

    private static byte[] encodeLong(long value) {
        return ByteBuffer.allocate(Long.BYTES).putLong(value).array();
    }

    private static long decodeLong(byte[] value) {
        return ByteBuffer.wrap(value).getLong();
    }

    private static final class SummingEngine implements TrainingEngine {

        private static final long serialVersionUID = 1L;

        private final int convergeAfter;
        private final int failPartition;

        private SummingEngine(int convergeAfter) {
            this(convergeAfter, -1);
        }

        private SummingEngine(int convergeAfter, int failPartition) {
            this.convergeAfter = convergeAfter;
            this.failPartition = failPartition;
        }

        @Override
        public byte[] initialize() {
            return encodeLong(0);
        }

        @Override
        public PartitionAccumulator newPartitionAccumulator(byte[] state, int partitionId) {
            if (partitionId == failPartition
                    && TaskContext.get() != null
                    && TaskContext.get().attemptNumber() == 0) {
                return new SummingAccumulator(true);
            }
            return new SummingAccumulator(false);
        }

        @Override
        public byte[] merge(byte[] state, List<PartitionPartial> partials) {
            long sum = 0L;
            int previousPartition = -1;
            for (PartitionPartial partial : partials) {
                if (partial.partitionId() <= previousPartition) {
                    throw new AssertionError("partials are not in deterministic order");
                }
                previousPartition = partial.partitionId();
                sum += decodeLong(partial.payload());
            }
            lastRoundSum = sum;
            return encodeLong(sum);
        }

        @Override
        public AdvanceResult advance(byte[] state, byte[] aggregate) {
            long iteration = decodeLong(state) + 1;
            return new AdvanceResult(encodeLong(iteration), iteration >= convergeAfter);
        }

        @Override
        public byte[] finish(byte[] state) {
            return encodeLong(lastRoundSum);
        }

        private long lastRoundSum;

        private static final class SummingAccumulator implements PartitionAccumulator {

            private long sum;
            private final boolean fail;

            private SummingAccumulator(boolean fail) {
                this.fail = fail;
            }

            @Override
            public void add(float[] vector) throws IOException {
                if (fail) {
                    throw new IOException("first task attempt fails");
                }
                sum += (long) vector[0];
            }

            @Override
            public byte[] finishPartial() {
                return encodeLong(sum);
            }

            @Override
            public void close() {}
        }
    }

    private static final class CloseTrackingEngine implements TrainingEngine {

        private static final long serialVersionUID = 1L;
        private boolean closed;

        @Override
        public byte[] initialize() {
            return encodeLong(0);
        }

        @Override
        public PartitionAccumulator newPartitionAccumulator(byte[] state, int partitionId) {
            return new PartitionAccumulator() {
                @Override
                public void add(float[] vector) throws IOException {
                    throw new IOException("injected failure");
                }

                @Override
                public byte[] finishPartial() {
                    return encodeLong(1);
                }

                @Override
                public void close() {
                    closed = true;
                }
            };
        }

        @Override
        public byte[] merge(byte[] state, List<PartitionPartial> partials) {
            throw new UnsupportedOperationException();
        }

        @Override
        public AdvanceResult advance(byte[] state, byte[] aggregate) {
            throw new UnsupportedOperationException();
        }

        @Override
        public byte[] finish(byte[] state) {
            throw new UnsupportedOperationException();
        }
    }

    private static final class BatchRecordingEngine implements TrainingEngine {

        private static final long serialVersionUID = 1L;
        private final List<Integer> batchSizes = new ArrayList<>();
        private final int maxVectorsPerBatch;

        private BatchRecordingEngine() {
            this(Integer.MAX_VALUE);
        }

        private BatchRecordingEngine(int maxVectorsPerBatch) {
            this.maxVectorsPerBatch = maxVectorsPerBatch;
        }

        @Override
        public byte[] initialize() {
            return encodeLong(0);
        }

        @Override
        public PartitionAccumulator newPartitionAccumulator(byte[] state, int partitionId) {
            return new PartitionAccumulator() {
                private long count;

                @Override
                public void add(float[] vector) {
                    throw new AssertionError("batch override must be used");
                }

                @Override
                public void addBatch(List<float[]> vectors) {
                    batchSizes.add(vectors.size());
                    count += vectors.size();
                }

                @Override
                public byte[] finishPartial() {
                    return encodeLong(count);
                }

                @Override
                public void close() {}
            };
        }

        @Override
        public int maxVectorsPerBatch() {
            return maxVectorsPerBatch;
        }

        @Override
        public byte[] merge(byte[] state, List<PartitionPartial> partials) {
            return partials.get(0).payload();
        }

        @Override
        public AdvanceResult advance(byte[] state, byte[] aggregate) {
            return new AdvanceResult(aggregate, true);
        }

        @Override
        public byte[] finish(byte[] state) {
            return state;
        }
    }

    private static final class ExactPartitioner extends org.apache.spark.Partitioner
            implements Serializable {

        private static final long serialVersionUID = 1L;
        private final int partitions;

        private ExactPartitioner(int partitions) {
            this.partitions = partitions;
        }

        @Override
        public int numPartitions() {
            return partitions;
        }

        @Override
        public int getPartition(Object key) {
            return (Integer) key;
        }
    }
}
