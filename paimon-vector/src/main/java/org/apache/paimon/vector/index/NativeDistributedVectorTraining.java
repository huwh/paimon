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

package org.apache.paimon.vector.index;

import org.apache.paimon.annotation.Experimental;
import org.apache.paimon.index.vector.DistributedVectorIndexTraining;
import org.apache.paimon.index.vector.VectorIndexTraining;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/** Experimental Paimon adapter for native distributed IVF-PQ training. */
@Experimental
public final class NativeDistributedVectorTraining {

    public static final int DEFAULT_MAX_PARTIAL_BYTES =
            (int) DistributedVectorIndexTraining.DEFAULT_MAX_PARTIAL_BYTES;

    private static final String INDEX_TYPE_OPTION = "index.type";
    private static final String DIMENSION_OPTION = "dimension";
    private static final String NLIST_OPTION = "nlist";
    private static final String METRIC_OPTION = "metric";
    private static final String USE_OPQ_OPTION = "use-opq";

    private NativeDistributedVectorTraining() {}

    /** Creates a versioned, serializable coarse-training state payload. */
    public static byte[] initializeCoarse(
            int dimension,
            int nlist,
            int maxIterations,
            double tolerance,
            float[] initialCentroids) {
        return DistributedVectorIndexTraining.initialize(
                        dimension, nlist, maxIterations, tolerance, initialCentroids)
                .serialized();
    }

    /** Creates a task-local streaming accumulator. The caller must close it. */
    public static CoarseAccumulator createCoarseAccumulator(byte[] state, long partitionId) {
        return new CoarseAccumulator(
                DistributedVectorIndexTraining.createAccumulator(restoreState(state), partitionId));
    }

    /** Creates a task-local accumulator with an explicit serialized-partial memory limit. */
    public static CoarseAccumulator createCoarseAccumulator(
            byte[] state, long partitionId, int maxPartialBytes) {
        return new CoarseAccumulator(
                DistributedVectorIndexTraining.createAccumulator(
                        restoreState(state), partitionId, maxPartialBytes));
    }

    /** Deterministically merges serialized partition partials. */
    public static byte[] mergeCoarse(byte[] state, List<byte[]> partials) {
        Objects.requireNonNull(partials, "partials");
        List<DistributedVectorIndexTraining.Partial> restored = new ArrayList<>(partials.size());
        for (int i = 0; i < partials.size(); i++) {
            restored.add(
                    DistributedVectorIndexTraining.restorePartial(
                            Objects.requireNonNull(partials.get(i), "partials[" + i + "]")));
        }
        return DistributedVectorIndexTraining.merge(restoreState(state), restored).serialized();
    }

    /** Advances the coarse-centroid state by one iteration. */
    public static CoarseIteration advanceCoarse(byte[] state, byte[] aggregate) {
        DistributedVectorIndexTraining.Iteration iteration =
                DistributedVectorIndexTraining.advance(
                        restoreState(state),
                        DistributedVectorIndexTraining.restoreAggregate(
                                requirePayload(aggregate, "aggregate")));
        return new CoarseIteration(iteration);
    }

    /** Finalizes a finished coarse-centroid state into a serializable model payload. */
    public static byte[] finalizeCoarse(byte[] state) {
        return DistributedVectorIndexTraining.finalizeModel(restoreState(state)).serialized();
    }

    /**
     * Trains the residual PQ model from a coarse model and a bounded raw sample.
     *
     * <p>Distributed coarse training currently supports only fixed-shape L2 IVF-PQ without OPQ.
     */
    public static VectorTrainingModel trainIvfPq(
            String indexType,
            Map<String, String> options,
            byte[] coarseModel,
            float[] rawSample,
            int vectorCount) {
        Map<String, String> validatedOptions = validateIvfPqOptions(indexType, options);
        int dimension = parseFixedPositiveOption(validatedOptions, DIMENSION_OPTION);
        int nlist = parseFixedPositiveOption(validatedOptions, NLIST_OPTION);
        validateRawSample(rawSample, vectorCount, dimension);
        DistributedVectorIndexTraining.Model model =
                DistributedVectorIndexTraining.restoreModel(
                        requirePayload(coarseModel, "coarseModel"));
        validateFixedShape(dimension, nlist, model);

        VectorIndexTraining training = null;
        try {
            training =
                    DistributedVectorIndexTraining.trainIvfPqFromCoarse(
                            validatedOptions, model, rawSample, vectorCount);
            VectorTrainingModel result =
                    NativeVectorGlobalModelTrainers.fromTraining(
                            indexType, validatedOptions, training);
            training = null;
            return result;
        } finally {
            if (training != null) {
                training.close();
            }
        }
    }

    private static DistributedVectorIndexTraining.State restoreState(byte[] state) {
        return DistributedVectorIndexTraining.restoreState(requirePayload(state, "state"));
    }

    private static byte[] requirePayload(byte[] payload, String name) {
        return Objects.requireNonNull(payload, name).clone();
    }

    private static Map<String, String> validateIvfPqOptions(
            String indexType, Map<String, String> options) {
        if (!NativeVectorIndexOptions.IVF_PQ_INDEX_TYPE.equals(indexType)) {
            throw new IllegalArgumentException(
                    "Distributed vector training supports only index type '"
                            + NativeVectorIndexOptions.IVF_PQ_INDEX_TYPE
                            + "', but was '"
                            + indexType
                            + "'.");
        }
        Objects.requireNonNull(options, "options");
        Map<String, String> copy = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : options.entrySet()) {
            copy.put(
                    Objects.requireNonNull(entry.getKey(), "options contains a null key"),
                    Objects.requireNonNull(entry.getValue(), "options[" + entry.getKey() + "]"));
        }

        String nativeIndexType = copy.get(INDEX_TYPE_OPTION);
        if (nativeIndexType != null && !"ivf_pq".equals(normalize(nativeIndexType))) {
            throw new IllegalArgumentException(
                    "Distributed vector training requires option 'index.type=ivf_pq'.");
        }
        copy.put(INDEX_TYPE_OPTION, "ivf_pq");

        String metric = copy.get(METRIC_OPTION);
        if (metric == null || !"l2".equals(normalize(metric))) {
            throw new IllegalArgumentException(
                    "Distributed vector training requires option 'metric=l2'.");
        }
        copy.put(METRIC_OPTION, "l2");

        String useOpq = copy.get(USE_OPQ_OPTION);
        if (useOpq != null && !"false".equals(normalize(useOpq))) {
            throw new IllegalArgumentException(
                    "Distributed vector training requires option 'use-opq=false'.");
        }
        copy.put(USE_OPQ_OPTION, "false");
        return copy;
    }

    private static void validateFixedShape(
            int dimension, int nlist, DistributedVectorIndexTraining.Model model) {
        if (dimension != model.dimension() || nlist != model.nlist()) {
            throw new IllegalArgumentException(
                    String.format(
                            "IVF-PQ options shape nlist=%d, dimension=%d does not match coarse "
                                    + "model shape nlist=%d, dimension=%d.",
                            nlist, dimension, model.nlist(), model.dimension()));
        }
    }

    private static void validateRawSample(float[] rawSample, int vectorCount, int dimension) {
        Objects.requireNonNull(rawSample, "rawSample");
        if (vectorCount <= 0) {
            throw new IllegalArgumentException("vectorCount must be positive.");
        }
        final int expectedLength;
        try {
            expectedLength = Math.multiplyExact(vectorCount, dimension);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException(
                    "vectorCount * dimension exceeds the maximum Java array length.", e);
        }
        if (rawSample.length != expectedLength) {
            throw new IllegalArgumentException(
                    "rawSample length must equal vectorCount * dimension ("
                            + expectedLength
                            + "), but was "
                            + rawSample.length
                            + ".");
        }
        for (int i = 0; i < rawSample.length; i++) {
            if (!Float.isFinite(rawSample[i])) {
                throw new IllegalArgumentException(
                        "rawSample contains a non-finite value at index " + i + ".");
            }
        }
    }

    private static int parseFixedPositiveOption(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null || "auto".equals(normalize(value))) {
            throw new IllegalArgumentException(
                    "Distributed vector training requires a fixed positive option '" + key + "'.");
        }
        final int parsed;
        try {
            parsed = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Distributed vector training requires a fixed positive option '"
                            + key
                            + "', but was '"
                            + value
                            + "'.",
                    e);
        }
        if (parsed <= 0) {
            throw new IllegalArgumentException(
                    "Distributed vector training requires a fixed positive option '"
                            + key
                            + "', but was '"
                            + value
                            + "'.");
        }
        return parsed;
    }

    private static String normalize(String value) {
        return value.trim().toLowerCase(Locale.ROOT).replace('-', '_');
    }

    /** Task-local native coarse-statistics accumulator. */
    public static final class CoarseAccumulator implements AutoCloseable {

        private final DistributedVectorIndexTraining.Accumulator accumulator;

        private CoarseAccumulator(DistributedVectorIndexTraining.Accumulator accumulator) {
            this.accumulator = accumulator;
        }

        public CoarseAccumulator addBatch(float[] vectors, int vectorCount) {
            accumulator.addBatch(Objects.requireNonNull(vectors, "vectors"), vectorCount);
            return this;
        }

        public byte[] finish() {
            return accumulator.finish().serialized();
        }

        @Override
        public void close() {
            accumulator.close();
        }
    }

    /** Serializable result of one coarse-centroid state transition. */
    public static final class CoarseIteration implements Serializable {

        private static final long serialVersionUID = 1L;

        private final byte[] state;
        private final int iteration;
        private final boolean converged;
        private final boolean finished;
        private final double maxCentroidShift;

        private CoarseIteration(DistributedVectorIndexTraining.Iteration iteration) {
            this.state = iteration.state().serialized();
            this.iteration = iteration.iteration();
            this.converged = iteration.converged();
            this.finished = iteration.finished();
            this.maxCentroidShift = iteration.maxCentroidShift();
        }

        public byte[] state() {
            return state.clone();
        }

        public int iteration() {
            return iteration;
        }

        public boolean converged() {
            return converged;
        }

        public boolean finished() {
            return finished;
        }

        public double maxCentroidShift() {
            return maxCentroidShift;
        }
    }
}
