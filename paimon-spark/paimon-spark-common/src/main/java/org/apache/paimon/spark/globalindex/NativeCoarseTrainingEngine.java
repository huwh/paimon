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
import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.AdvanceResult;
import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.PartitionAccumulator;
import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.PartitionPartial;
import org.apache.paimon.spark.globalindex.SparkVectorTrainingOrchestrator.TrainingEngine;
import org.apache.paimon.vector.index.NativeDistributedVectorTraining;

import java.util.ArrayList;
import java.util.List;

import static org.apache.paimon.utils.Preconditions.checkArgument;
import static org.apache.paimon.utils.Preconditions.checkNotNull;

/** Adapter from the Spark iterative state machine to native distributed coarse training. */
@Experimental
public final class NativeCoarseTrainingEngine implements TrainingEngine {

    private static final long serialVersionUID = 1L;
    private static final int MAX_NATIVE_BATCH_BYTES = 4 * 1024 * 1024;

    private final int dimension;
    private final int nlist;
    private final int maxIterations;
    private final double tolerance;
    private final int maxPartialBytes;
    // Executors need only dimension and the broadcast state. Do not copy bootstrap data into every
    // task closure after the driver has initialized the state.
    private final transient float[] initialCentroids;

    public NativeCoarseTrainingEngine(
            int dimension,
            int nlist,
            int maxIterations,
            double tolerance,
            float[] initialCentroids) {
        this(
                dimension,
                nlist,
                maxIterations,
                tolerance,
                initialCentroids,
                NativeDistributedVectorTraining.DEFAULT_MAX_PARTIAL_BYTES);
    }

    public NativeCoarseTrainingEngine(
            int dimension,
            int nlist,
            int maxIterations,
            double tolerance,
            float[] initialCentroids,
            int maxPartialBytes) {
        checkArgument(dimension > 0, "Training vector dimension must be greater than 0.");
        checkArgument(nlist > 0, "Training nlist must be greater than 0.");
        checkArgument(maxIterations > 0, "Training max iterations must be greater than 0.");
        checkArgument(
                !Double.isNaN(tolerance) && !Double.isInfinite(tolerance) && tolerance >= 0.0d,
                "Training tolerance must be finite and non-negative.");
        checkNotNull(initialCentroids, "initialCentroids");
        checkArgument(
                (long) dimension * nlist <= Integer.MAX_VALUE
                        && initialCentroids.length == dimension * nlist,
                "Initial centroids length must equal dimension * nlist.");
        for (int i = 0; i < initialCentroids.length; i++) {
            checkArgument(
                    !Float.isNaN(initialCentroids[i]) && !Float.isInfinite(initialCentroids[i]),
                    "Initial centroids contain a non-finite value at index %s.",
                    i);
        }
        checkArgument(maxPartialBytes > 0, "Training max partial bytes must be greater than 0.");
        this.dimension = dimension;
        this.nlist = nlist;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.maxPartialBytes = maxPartialBytes;
        this.initialCentroids = initialCentroids.clone();
    }

    @Override
    public byte[] initialize() {
        checkNotNull(
                initialCentroids,
                "Initial centroids are unavailable after executor serialization.");
        return NativeDistributedVectorTraining.initializeCoarse(
                dimension, nlist, maxIterations, tolerance, initialCentroids);
    }

    @Override
    public PartitionAccumulator newPartitionAccumulator(byte[] state, int partitionId) {
        NativeDistributedVectorTraining.CoarseAccumulator accumulator =
                NativeDistributedVectorTraining.createCoarseAccumulator(
                        state, partitionId, maxPartialBytes);
        return new NativePartitionAccumulator(accumulator, dimension);
    }

    @Override
    public int maxVectorsPerBatch() {
        return maxVectorsForDimension(dimension);
    }

    @Override
    public byte[] merge(byte[] state, List<PartitionPartial> partials) {
        List<byte[]> payloads = new ArrayList<>(partials.size());
        for (PartitionPartial partial : partials) {
            payloads.add(partial.payload());
        }
        return NativeDistributedVectorTraining.mergeCoarse(state, payloads);
    }

    @Override
    public AdvanceResult advance(byte[] state, byte[] aggregate) {
        NativeDistributedVectorTraining.CoarseIteration iteration =
                NativeDistributedVectorTraining.advanceCoarse(state, aggregate);
        return new AdvanceResult(iteration.state(), iteration.converged(), iteration.finished());
    }

    @Override
    public byte[] finish(byte[] state) {
        return NativeDistributedVectorTraining.finalizeCoarse(state);
    }

    private static final class NativePartitionAccumulator implements PartitionAccumulator {

        private final NativeDistributedVectorTraining.CoarseAccumulator delegate;
        private final int dimension;

        private NativePartitionAccumulator(
                NativeDistributedVectorTraining.CoarseAccumulator delegate, int dimension) {
            this.delegate = checkNotNull(delegate, "delegate");
            this.dimension = dimension;
        }

        @Override
        public void add(float[] vector) {
            checkVector(vector);
            delegate.addBatch(vector, 1);
        }

        @Override
        public void addBatch(List<float[]> vectors) {
            checkNotNull(vectors, "vectors");
            checkArgument(
                    vectors.size() <= maxVectorsForDimension(dimension),
                    "Training batch has %s vectors, exceeding the native batch limit %s.",
                    vectors.size(),
                    maxVectorsForDimension(dimension));
            float[] batch = new float[Math.multiplyExact(vectors.size(), dimension)];
            for (int i = 0; i < vectors.size(); i++) {
                float[] vector = vectors.get(i);
                checkVector(vector);
                System.arraycopy(vector, 0, batch, i * dimension, dimension);
            }
            delegate.addBatch(batch, vectors.size());
        }

        @Override
        public byte[] finishPartial() {
            return delegate.finish();
        }

        @Override
        public void close() {
            delegate.close();
        }

        private void checkVector(float[] vector) {
            checkNotNull(vector, "vector");
            checkArgument(
                    vector.length == dimension,
                    "Training vector dimension %s does not match configured dimension %s.",
                    vector.length,
                    dimension);
        }
    }

    private static int maxVectorsForDimension(int dimension) {
        return (int) Math.max(1L, MAX_NATIVE_BATCH_BYTES / ((long) dimension * Float.BYTES));
    }
}
