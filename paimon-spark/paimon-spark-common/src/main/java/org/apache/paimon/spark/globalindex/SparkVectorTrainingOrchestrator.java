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

import org.apache.paimon.annotation.Experimental;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import java.io.Closeable;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import static org.apache.paimon.utils.Preconditions.checkArgument;
import static org.apache.paimon.utils.Preconditions.checkNotNull;

/** Driver-side state machine for iterative local or Spark vector training phases. */
@Experimental
public final class SparkVectorTrainingOrchestrator {

    public static final int DEFAULT_MAX_ITERATIONS = 20;
    public static final int MAX_ITERATIONS_LIMIT = 10_000;
    public static final int DEFAULT_MAX_BOOTSTRAP_BYTES = 64 * 1024 * 1024;
    public static final int DEFAULT_MAX_STATE_BYTES = 256 * 1024 * 1024;
    public static final int DEFAULT_MAX_PARTIAL_BYTES = 64 * 1024 * 1024;
    public static final long DEFAULT_MAX_DRIVER_PARTIAL_BYTES = 256L * 1024 * 1024;
    public static final int DEFAULT_PARTITION_BATCH_SIZE = 1024;
    public static final int MAX_PARTITION_BATCH_SIZE = 1_000_000;

    private SparkVectorTrainingOrchestrator() {}

    /**
     * Runs a training phase over a Spark RDD.
     *
     * <p>Every iteration is one RDD action. Spark therefore exposes only the winning attempt for
     * each logical partition to {@code collect}; failed or speculative attempts cannot add their
     * partial twice. The driver additionally canonicalizes by logical partition id before merge.
     */
    public static TrainingResult trainDistributed(
            JavaRDD<float[]> vectors, TrainingEngine engine, Configuration configuration)
            throws IOException {
        checkNotNull(vectors, "vectors");
        return train(engine, configuration, new SparkRoundExecutor(vectors));
    }

    /**
     * Runs the same state machine locally. Each iterable must return a fresh iterator on every
     * invocation because iterative training scans every partition once per round.
     */
    public static TrainingResult trainLocal(
            List<? extends Iterable<float[]>> partitions,
            TrainingEngine engine,
            Configuration configuration)
            throws IOException {
        checkNotNull(partitions, "partitions");
        return train(engine, configuration, new LocalRoundExecutor(partitions));
    }

    static TrainingResult train(
            TrainingEngine engine, Configuration configuration, RoundExecutor executor)
            throws IOException {
        checkNotNull(engine, "engine");
        checkNotNull(configuration, "configuration");
        checkNotNull(executor, "executor");

        byte[] state =
                requireBytesWithin(
                        engine.initialize(),
                        "initial training state",
                        configuration.maxStateBytes());
        long vectorCount = -1L;
        int iterations = 0;
        boolean converged = false;

        while (iterations < configuration.maxIterations()) {
            int effectiveMaxPartialBytes =
                    effectiveMaxPartialBytes(configuration, executor.numPartitions());
            int effectivePartitionBatchSize =
                    Math.min(configuration.partitionBatchSize(), engine.maxVectorsPerBatch());
            checkArgument(
                    effectivePartitionBatchSize > 0,
                    "Training engine max vectors per batch must be greater than 0.");
            List<PartitionPartial> collected =
                    executor.execute(
                            state, engine, effectivePartitionBatchSize, effectiveMaxPartialBytes);
            List<PartitionPartial> partials =
                    canonicalizePartials(
                            collected,
                            executor.numPartitions(),
                            configuration.maxPartialBytes(),
                            configuration.maxDriverPartialBytes());
            RoundStatistics statistics = roundStatistics(partials);
            if (statistics.vectorCount == 0) {
                throw new IllegalArgumentException(
                        "Vector training requires at least one vector, but every partition was empty.");
            }
            if (vectorCount < 0) {
                vectorCount = statistics.vectorCount;
            } else if (vectorCount != statistics.vectorCount) {
                throw new IllegalStateException(
                        "Vector training input changed between iterations: expected "
                                + vectorCount
                                + " vectors, but found "
                                + statistics.vectorCount
                                + '.');
            }

            byte[] aggregate =
                    requireBytesWithin(
                            engine.merge(state.clone(), statistics.nonEmptyPartials),
                            "training aggregate",
                            configuration.maxDriverPartialBytes());
            AdvanceResult advance =
                    checkNotNull(
                            engine.advance(state.clone(), aggregate),
                            "Training engine returned a null advance result.");
            state =
                    requireBytesWithin(
                            advance.state(),
                            "advanced training state",
                            configuration.maxStateBytes());
            converged = advance.converged();
            iterations++;
            if (advance.finished()) {
                break;
            }
        }

        byte[] model =
                requireBytesWithin(
                        engine.finish(state.clone()),
                        "training model",
                        configuration.maxStateBytes());
        return new TrainingResult(
                model, state, iterations, converged, vectorCount, executor.numPartitions());
    }

    /**
     * Selects the first {@code nlist} vectors in stable Spark partition and row order. The caller
     * must build the RDD from a deterministic split order. Memory is bounded by the centroid state
     * itself ({@code nlist * dimension} floats).
     */
    public static float[] deterministicInitialCentroids(
            JavaRDD<float[]> vectors, int dimension, int nlist) {
        return deterministicInitialCentroids(
                vectors, dimension, nlist, DEFAULT_MAX_BOOTSTRAP_BYTES);
    }

    public static float[] deterministicInitialCentroids(
            JavaRDD<float[]> vectors, int dimension, int nlist, long maxBootstrapBytes) {
        checkNotNull(vectors, "vectors");
        validateBootstrapShape(dimension, nlist, maxBootstrapBytes);
        return flattenInitialCentroids(
                vectors.map(vector -> validateBootstrapVector(vector, dimension)).take(nlist),
                dimension,
                nlist);
    }

    /** Selects deterministic initial centroids from a stable local iteration order. */
    public static float[] deterministicInitialCentroids(
            Iterable<float[]> vectors, int dimension, int nlist) {
        return deterministicInitialCentroids(
                vectors, dimension, nlist, DEFAULT_MAX_BOOTSTRAP_BYTES);
    }

    public static float[] deterministicInitialCentroids(
            Iterable<float[]> vectors, int dimension, int nlist, long maxBootstrapBytes) {
        checkNotNull(vectors, "vectors");
        validateBootstrapShape(dimension, nlist, maxBootstrapBytes);
        List<float[]> selected = new ArrayList<>(nlist);
        Iterator<float[]> iterator = vectors.iterator();
        while (selected.size() < nlist && iterator.hasNext()) {
            selected.add(iterator.next());
        }
        return flattenInitialCentroids(selected, dimension, nlist);
    }

    static List<PartitionPartial> canonicalizePartials(
            List<PartitionPartial> partials,
            int expectedPartitions,
            int maxPartialBytes,
            long maxDriverPartialBytes) {
        checkNotNull(partials, "partials");
        checkArgument(expectedPartitions >= 0, "Expected partition count must not be negative.");
        checkArgument(maxPartialBytes > 0, "Training max partial bytes must be greater than 0.");
        checkArgument(
                maxDriverPartialBytes > 0,
                "Training max driver partial bytes must be greater than 0.");

        List<PartitionPartial> sorted = new ArrayList<>(partials.size());
        for (PartitionPartial partial : partials) {
            sorted.add(checkNotNull(partial, "A training partition returned a null partial."));
        }
        sorted.sort(Comparator.comparingInt(PartitionPartial::partitionId));
        List<PartitionPartial> canonical = new ArrayList<>(sorted.size());
        long totalPayloadBytes = 0L;
        PartitionPartial previous = null;
        for (PartitionPartial partial : sorted) {
            int partitionId = partial.partitionId();
            checkArgument(
                    partitionId >= 0 && partitionId < expectedPartitions,
                    "Training partial partition id %s is outside [0, %s).",
                    partitionId,
                    expectedPartitions);
            checkArgument(
                    partial.payloadSize() <= maxPartialBytes,
                    "Training partial for partition %s is %s bytes, exceeding the %s byte limit.",
                    partitionId,
                    partial.payloadSize(),
                    maxPartialBytes);

            if (previous != null && previous.partitionId() == partitionId) {
                if (!previous.sameResult(partial)) {
                    throw new IllegalStateException(
                            "Conflicting training partials for logical partition "
                                    + partitionId
                                    + ". Retried attempts must produce identical results.");
                }
                // A retry/speculative duplicate must never be merged or counted twice.
                continue;
            }

            totalPayloadBytes = saturatedAdd(totalPayloadBytes, partial.payloadSize());
            checkArgument(
                    totalPayloadBytes <= maxDriverPartialBytes,
                    "Training partials require %s driver bytes, exceeding the %s byte limit.",
                    totalPayloadBytes,
                    maxDriverPartialBytes);
            canonical.add(partial);
            previous = partial;
        }

        for (int partitionId = 0; partitionId < expectedPartitions; partitionId++) {
            if (partitionId >= canonical.size()
                    || canonical.get(partitionId).partitionId() != partitionId) {
                throw new IllegalStateException(
                        "Missing training partial for logical partition " + partitionId + '.');
            }
        }
        return canonical;
    }

    private static RoundStatistics roundStatistics(List<PartitionPartial> partials) {
        long vectorCount = 0L;
        List<PartitionPartial> nonEmpty = new ArrayList<>();
        for (PartitionPartial partial : partials) {
            vectorCount = saturatedAdd(vectorCount, partial.vectorCount());
            if (!partial.isEmpty()) {
                nonEmpty.add(partial);
            }
        }
        return new RoundStatistics(vectorCount, Collections.unmodifiableList(nonEmpty));
    }

    private static PartitionPartial accumulatePartition(
            TrainingEngine engine,
            byte[] state,
            int partitionId,
            Iterator<float[]> vectors,
            int partitionBatchSize,
            int maxPartialBytes)
            throws IOException {
        if (!vectors.hasNext()) {
            return PartitionPartial.empty(partitionId);
        }

        long vectorCount = 0L;
        List<float[]> batch = new ArrayList<>(partitionBatchSize);
        try (PartitionAccumulator accumulator =
                checkNotNull(
                        engine.newPartitionAccumulator(state.clone(), partitionId),
                        "Training engine returned a null partition accumulator.")) {
            do {
                float[] vector = vectors.next();
                if (vector == null) {
                    throw new IllegalArgumentException(
                            "Training vector must not be null in partition "
                                    + partitionId
                                    + " at offset "
                                    + vectorCount
                                    + '.');
                }
                batch.add(vector);
                if (vectorCount == Long.MAX_VALUE) {
                    throw new IllegalStateException(
                            "Training vector count overflow in partition " + partitionId + '.');
                }
                vectorCount++;
                if (batch.size() == partitionBatchSize) {
                    accumulator.addBatch(batch);
                    batch.clear();
                }
            } while (vectors.hasNext());
            if (!batch.isEmpty()) {
                accumulator.addBatch(batch);
            }
            return new PartitionPartial(
                    partitionId,
                    vectorCount,
                    requireBytesWithin(
                            accumulator.finishPartial(),
                            "training partial for partition " + partitionId,
                            maxPartialBytes));
        }
    }

    private static int effectiveMaxPartialBytes(Configuration configuration, int partitionCount) {
        checkArgument(partitionCount >= 0, "Vector training partition count must not be negative.");
        if (partitionCount == 0) {
            return configuration.maxPartialBytes();
        }
        long bytesPerPartition = configuration.maxDriverPartialBytes() / partitionCount;
        checkArgument(
                bytesPerPartition > 0,
                "Training has %s partitions, exceeding the %s byte driver partial budget.",
                partitionCount,
                configuration.maxDriverPartialBytes());
        return (int)
                Math.min(
                        configuration.maxPartialBytes(),
                        Math.min(bytesPerPartition, Integer.MAX_VALUE));
    }

    private static float[] flattenInitialCentroids(
            List<float[]> selected, int dimension, int nlist) {
        if (selected.size() != nlist) {
            throw new IllegalArgumentException(
                    "Distributed vector training requires at least nlist="
                            + nlist
                            + " non-null vectors for deterministic bootstrap, but found "
                            + selected.size()
                            + '.');
        }
        float[] centroids = new float[Math.multiplyExact(dimension, nlist)];
        for (int centroid = 0; centroid < nlist; centroid++) {
            float[] vector = selected.get(centroid);
            checkNotNull(vector, "Bootstrap vector %s must not be null.", centroid);
            checkArgument(
                    vector.length == dimension,
                    "Bootstrap vector %s has dimension %s, expected %s.",
                    centroid,
                    vector.length,
                    dimension);
            for (int d = 0; d < dimension; d++) {
                checkArgument(
                        !Float.isNaN(vector[d]) && !Float.isInfinite(vector[d]),
                        "Bootstrap vector %s contains a non-finite value at dimension %s.",
                        centroid,
                        d);
            }
            System.arraycopy(vector, 0, centroids, centroid * dimension, dimension);
        }
        return centroids;
    }

    private static float[] validateBootstrapVector(float[] vector, int dimension) {
        checkNotNull(vector, "A bootstrap vector must not be null.");
        checkArgument(
                vector.length == dimension,
                "A bootstrap vector has dimension %s, expected %s.",
                vector.length,
                dimension);
        return vector;
    }

    private static void validateBootstrapShape(int dimension, int nlist, long maxBootstrapBytes) {
        checkArgument(dimension > 0, "Training vector dimension must be greater than 0.");
        checkArgument(nlist > 0, "Training nlist must be greater than 0.");
        checkArgument(
                maxBootstrapBytes > 0, "Training max bootstrap bytes must be greater than 0.");
        long centroidValues = (long) dimension * nlist;
        checkArgument(
                centroidValues <= Integer.MAX_VALUE,
                "Training centroid bootstrap has too many values for dimension=%s and nlist=%s.",
                dimension,
                nlist);
        long centroidBytes = centroidValues * Float.BYTES;
        checkArgument(
                centroidBytes <= maxBootstrapBytes,
                "Training centroid bootstrap requires %s bytes for dimension=%s and nlist=%s, "
                        + "exceeding the %s byte limit.",
                centroidBytes,
                dimension,
                nlist,
                maxBootstrapBytes);
    }

    private static byte[] requireBytes(byte[] bytes, String description) {
        return checkNotNull(bytes, "Training engine returned null for %s.", description).clone();
    }

    private static byte[] requireBytesWithin(byte[] bytes, String description, long maxBytes) {
        checkNotNull(bytes, "Training engine returned null for %s.", description);
        checkArgument(
                bytes.length <= maxBytes,
                "The %s is %s bytes, exceeding the %s byte limit.",
                description,
                bytes.length,
                maxBytes);
        return bytes.clone();
    }

    private static long saturatedAdd(long left, long right) {
        long result = left + right;
        if (((left ^ result) & (right ^ result)) < 0) {
            return Long.MAX_VALUE;
        }
        return result;
    }

    /** Versioned state/partial codec and algorithm adapter for one training phase. */
    public interface TrainingEngine extends Serializable {

        /** Creates the initial serialized state, including any deterministic bootstrap data. */
        byte[] initialize() throws IOException;

        /**
         * Creates a streaming accumulator for one logical input partition and one state version.
         */
        PartitionAccumulator newPartitionAccumulator(byte[] state, int partitionId)
                throws IOException;

        /** Caps vectors retained by the orchestrator before one synchronous batch call. */
        default int maxVectorsPerBatch() {
            return Integer.MAX_VALUE;
        }

        /** Deterministically merges non-empty partials sorted by logical partition id. */
        byte[] merge(byte[] state, List<PartitionPartial> partials) throws IOException;

        /** Advances the driver state by exactly one iteration. */
        AdvanceResult advance(byte[] state, byte[] aggregate) throws IOException;

        /** Finalizes this phase from the last driver state. */
        byte[] finish(byte[] state) throws IOException;
    }

    /** Executor-local, streaming partition accumulator. */
    public interface PartitionAccumulator extends Closeable {

        /** Adds one vector without retaining the input array after this call returns. */
        void add(float[] vector) throws IOException;

        /**
         * Consumes one bounded batch synchronously. Implementations may override this method to use
         * a native batch API, but must not retain the list or its arrays after returning.
         */
        default void addBatch(List<float[]> vectors) throws IOException {
            for (float[] vector : vectors) {
                add(vector);
            }
        }

        /**
         * Produces a bounded sufficient-statistics payload. K-means engines should encode only
         * touched clusters instead of a dense {@code nlist * dimension} matrix.
         */
        byte[] finishPartial() throws IOException;

        @Override
        void close() throws IOException;
    }

    interface RoundExecutor {

        int numPartitions();

        List<PartitionPartial> execute(
                byte[] state, TrainingEngine engine, int partitionBatchSize, int maxPartialBytes)
                throws IOException;
    }

    /** Serialized output from exactly one logical input partition. */
    public static final class PartitionPartial implements Serializable {

        private static final long serialVersionUID = 1L;

        private final int partitionId;
        private final long vectorCount;
        private final byte[] payload;

        public PartitionPartial(int partitionId, long vectorCount, byte[] payload) {
            checkArgument(partitionId >= 0, "Partition id must not be negative.");
            checkArgument(vectorCount >= 0, "Partial vector count must not be negative.");
            checkNotNull(payload, "payload");
            checkArgument(
                    vectorCount != 0 || payload.length == 0,
                    "An empty partition partial must not contain a payload.");
            this.partitionId = partitionId;
            this.vectorCount = vectorCount;
            this.payload = payload.clone();
        }

        public static PartitionPartial empty(int partitionId) {
            return new PartitionPartial(partitionId, 0L, new byte[0]);
        }

        public int partitionId() {
            return partitionId;
        }

        public long vectorCount() {
            return vectorCount;
        }

        public byte[] payload() {
            return payload.clone();
        }

        public boolean isEmpty() {
            return vectorCount == 0;
        }

        private int payloadSize() {
            return payload.length;
        }

        private boolean sameResult(PartitionPartial other) {
            return vectorCount == other.vectorCount && Arrays.equals(payload, other.payload);
        }
    }

    /** Result of one driver-side state transition. */
    public static final class AdvanceResult {

        private final byte[] state;
        private final boolean converged;
        private final boolean finished;

        public AdvanceResult(byte[] state, boolean converged) {
            this(state, converged, converged);
        }

        public AdvanceResult(byte[] state, boolean converged, boolean finished) {
            checkArgument(!converged || finished, "A converged training state must be finished.");
            this.state = checkNotNull(state, "state").clone();
            this.converged = converged;
            this.finished = finished;
        }

        public byte[] state() {
            return state.clone();
        }

        public boolean converged() {
            return converged;
        }

        public boolean finished() {
            return finished;
        }
    }

    /** Iteration and driver-memory bounds for one training phase. */
    public static final class Configuration {

        private final int maxIterations;
        private final int maxStateBytes;
        private final int maxPartialBytes;
        private final long maxDriverPartialBytes;
        private final int partitionBatchSize;

        public Configuration(int maxIterations, int maxPartialBytes, long maxDriverPartialBytes) {
            this(
                    maxIterations,
                    DEFAULT_MAX_STATE_BYTES,
                    maxPartialBytes,
                    maxDriverPartialBytes,
                    DEFAULT_PARTITION_BATCH_SIZE);
        }

        public Configuration(
                int maxIterations,
                int maxStateBytes,
                int maxPartialBytes,
                long maxDriverPartialBytes) {
            this(
                    maxIterations,
                    maxStateBytes,
                    maxPartialBytes,
                    maxDriverPartialBytes,
                    DEFAULT_PARTITION_BATCH_SIZE);
        }

        public Configuration(
                int maxIterations,
                int maxStateBytes,
                int maxPartialBytes,
                long maxDriverPartialBytes,
                int partitionBatchSize) {
            checkArgument(
                    maxIterations > 0 && maxIterations <= MAX_ITERATIONS_LIMIT,
                    "Training max iterations must be in [1, %s], but was %s.",
                    MAX_ITERATIONS_LIMIT,
                    maxIterations);
            checkArgument(maxStateBytes > 0, "Training max state bytes must be greater than 0.");
            checkArgument(
                    maxPartialBytes > 0, "Training max partial bytes must be greater than 0.");
            checkArgument(
                    maxDriverPartialBytes > 0,
                    "Training max driver partial bytes must be greater than 0.");
            checkArgument(
                    partitionBatchSize > 0 && partitionBatchSize <= MAX_PARTITION_BATCH_SIZE,
                    "Training partition batch size must be in [1, %s], but was %s.",
                    MAX_PARTITION_BATCH_SIZE,
                    partitionBatchSize);
            this.maxIterations = maxIterations;
            this.maxStateBytes = maxStateBytes;
            this.maxPartialBytes = maxPartialBytes;
            this.maxDriverPartialBytes = maxDriverPartialBytes;
            this.partitionBatchSize = partitionBatchSize;
        }

        public static Configuration defaults() {
            return new Configuration(
                    DEFAULT_MAX_ITERATIONS,
                    DEFAULT_MAX_STATE_BYTES,
                    DEFAULT_MAX_PARTIAL_BYTES,
                    DEFAULT_MAX_DRIVER_PARTIAL_BYTES,
                    DEFAULT_PARTITION_BATCH_SIZE);
        }

        public int maxIterations() {
            return maxIterations;
        }

        public int maxPartialBytes() {
            return maxPartialBytes;
        }

        public int maxStateBytes() {
            return maxStateBytes;
        }

        public long maxDriverPartialBytes() {
            return maxDriverPartialBytes;
        }

        public int partitionBatchSize() {
            return partitionBatchSize;
        }
    }

    /** Final output and bounded diagnostics for one training phase. */
    public static final class TrainingResult {

        private final byte[] model;
        private final byte[] finalState;
        private final int iterations;
        private final boolean converged;
        private final long vectorCount;
        private final int partitionCount;

        private TrainingResult(
                byte[] model,
                byte[] finalState,
                int iterations,
                boolean converged,
                long vectorCount,
                int partitionCount) {
            this.model = model.clone();
            this.finalState = finalState.clone();
            this.iterations = iterations;
            this.converged = converged;
            this.vectorCount = vectorCount;
            this.partitionCount = partitionCount;
        }

        public byte[] model() {
            return model.clone();
        }

        public byte[] finalState() {
            return finalState.clone();
        }

        public int iterations() {
            return iterations;
        }

        public boolean converged() {
            return converged;
        }

        public long vectorCount() {
            return vectorCount;
        }

        public int partitionCount() {
            return partitionCount;
        }
    }

    private static final class RoundStatistics {

        private final long vectorCount;
        private final List<PartitionPartial> nonEmptyPartials;

        private RoundStatistics(long vectorCount, List<PartitionPartial> nonEmptyPartials) {
            this.vectorCount = vectorCount;
            this.nonEmptyPartials = nonEmptyPartials;
        }
    }

    private static final class SparkRoundExecutor implements RoundExecutor {

        private final JavaRDD<float[]> vectors;

        private SparkRoundExecutor(JavaRDD<float[]> vectors) {
            this.vectors = vectors;
        }

        @Override
        public int numPartitions() {
            return vectors.getNumPartitions();
        }

        @Override
        public List<PartitionPartial> execute(
                byte[] state, TrainingEngine engine, int partitionBatchSize, int maxPartialBytes) {
            JavaSparkContext context = JavaSparkContext.fromSparkContext(vectors.context());
            Broadcast<byte[]> stateBroadcast = context.broadcast(state.clone());
            try {
                // One action owns one round. Spark task retries replace, rather than append to, a
                // logical partition result. No accumulator or driver-side mutable counter is used.
                return vectors.mapPartitionsWithIndex(
                                (partitionId, partition) ->
                                        Collections.singletonList(
                                                        accumulatePartition(
                                                                engine,
                                                                stateBroadcast.value(),
                                                                partitionId,
                                                                partition,
                                                                partitionBatchSize,
                                                                maxPartialBytes))
                                                .iterator(),
                                true)
                        .collect();
            } finally {
                stateBroadcast.destroy(false);
            }
        }
    }

    private static final class LocalRoundExecutor implements RoundExecutor {

        private final List<? extends Iterable<float[]>> partitions;

        private LocalRoundExecutor(List<? extends Iterable<float[]>> partitions) {
            this.partitions = new ArrayList<>(partitions);
            for (Iterable<float[]> partition : this.partitions) {
                checkNotNull(partition, "A local training partition must not be null.");
            }
        }

        @Override
        public int numPartitions() {
            return partitions.size();
        }

        @Override
        public List<PartitionPartial> execute(
                byte[] state, TrainingEngine engine, int partitionBatchSize, int maxPartialBytes)
                throws IOException {
            List<PartitionPartial> partials = new ArrayList<>(partitions.size());
            for (int partitionId = 0; partitionId < partitions.size(); partitionId++) {
                partials.add(
                        accumulatePartition(
                                engine,
                                state,
                                partitionId,
                                partitions.get(partitionId).iterator(),
                                partitionBatchSize,
                                maxPartialBytes));
            }
            return partials;
        }
    }
}
