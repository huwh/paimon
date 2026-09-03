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

import org.apache.paimon.data.InternalArray;
import org.apache.paimon.data.InternalVector;
import org.apache.paimon.globalindex.GlobalIndexSingleColumnWriter;
import org.apache.paimon.globalindex.GlobalIndexWriter;
import org.apache.paimon.globalindex.ResultEntry;
import org.apache.paimon.globalindex.io.GlobalIndexFileWriter;
import org.apache.paimon.index.vector.VectorIndexTrainer;
import org.apache.paimon.index.vector.VectorIndexTraining;
import org.apache.paimon.index.vector.VectorIndexWriter;

import javax.annotation.Nullable;

import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Factory for native global model trainers. */
public class NativeVectorGlobalModelTrainers {

    private static final int TRAIN_BATCH_SIZE = 4096;

    private NativeVectorGlobalModelTrainers() {}

    public static GlobalIndexWriter create(String indexType, Map<String, String> options) {
        return new NativeVectorTrainingModelWriter(indexType, options);
    }

    public static VectorTrainingModel load(
            String indexType, Map<String, String> options, byte[] payload) throws IOException {
        return NativeVectorTrainingModel.load(indexType, options, payload.clone());
    }

    static VectorTrainingModel fromTraining(
            String indexType, Map<String, String> options, VectorIndexTraining training) {
        return new NativeVectorTrainingModel(indexType, options, training);
    }

    /**
     * Java-side trainer for the centroid global IVF/PQ model.
     *
     * <p>This class is intentionally not a placeholder. It owns the same Java-side work as {@link
     * NativeVectorGlobalIndexWriter}: validate/materialize vectors, batch them, feed the existing
     * native {@link VectorIndexTrainer}, and convert the native training result into a Paimon
     * {@link VectorTrainingModel}. Native model persistence, routing, and centroid shard writing
     * use the public paimon-vector-index Java API directly.
     */
    private static class NativeVectorTrainingModelWriter
            implements VectorTrainingModelWriter, Closeable {

        private final String indexType;
        private final Map<String, String> options;
        private final int dim;
        private final VectorIndexTrainer trainer;
        private final float[] vectorBuf;
        private final int trainBatchSize;
        private float[] batchVectors;
        private int batchCount;
        private long count;
        private boolean finished;

        private NativeVectorTrainingModelWriter(String indexType, Map<String, String> options) {
            this.indexType = indexType;
            this.options = new LinkedHashMap<>(options);
            this.dim = parseDimension(options);
            this.trainer = VectorIndexTrainer.create(options);
            this.vectorBuf = new float[dim];
            this.trainBatchSize =
                    NativeVectorGlobalIndexWriter.vectorBatchSize(TRAIN_BATCH_SIZE, dim);
            this.batchVectors = new float[trainBatchSize * dim];
        }

        @Override
        public void write(@Nullable Object key, long absoluteRowId) {
            ensureNotFinished();
            if (key == null) {
                return;
            }
            float[] vector = materializeAndValidate(key, absoluteRowId);
            System.arraycopy(vector, 0, batchVectors, batchCount * dim, dim);
            batchCount++;
            count++;
            if (batchCount == trainBatchSize) {
                try {
                    flushTrainingBatch();
                } catch (IOException e) {
                    throw new RuntimeException("Failed to flush native vector training batch.", e);
                }
            }
        }

        @Override
        public VectorTrainingModel finishTraining() throws IOException {
            ensureNotFinished();
            finished = true;
            if (count == 0) {
                close();
                throw new IllegalStateException(
                        "Native global model trainer for '"
                                + indexType
                                + "' requires at least one non-null training vector.");
            }

            try {
                flushTrainingBatch();
                VectorIndexTraining training = trainer.finishTraining();
                return new NativeVectorTrainingModel(indexType, options, training);
            } finally {
                batchVectors = null;
                closeTrainerQuietly();
            }
        }

        @Override
        public List<ResultEntry> finish() {
            throw new UnsupportedOperationException(
                    "Native global model trainer for '"
                            + indexType
                            + "' is used only by centroid build and produces a "
                            + "VectorTrainingModel via finishTraining(). It must not write direct "
                            + "ResultEntry files via finish().");
        }

        @Override
        public void close() throws IOException {
            batchVectors = null;
            closeTrainerQuietly();
        }

        private void flushTrainingBatch() throws IOException {
            if (batchCount == 0) {
                return;
            }
            if (batchCount == trainBatchSize) {
                trainer.addTrainingVectors(batchVectors, batchCount);
            } else {
                trainer.addTrainingVectors(
                        Arrays.copyOf(batchVectors, batchCount * dim), batchCount);
            }
            batchCount = 0;
        }

        private float[] materializeAndValidate(Object fieldData, long absoluteRowId) {
            if (fieldData instanceof float[]) {
                float[] vector = (float[]) fieldData;
                checkDimension(vector.length);
                for (int i = 0; i < dim; i++) {
                    checkFinite(vector[i], absoluteRowId, i);
                }
                return vector;
            } else if (fieldData instanceof InternalVector) {
                InternalVector vector = (InternalVector) fieldData;
                checkDimension(vector.size());
                for (int i = 0; i < dim; i++) {
                    float v = vector.getFloat(i);
                    checkFinite(v, absoluteRowId, i);
                    vectorBuf[i] = v;
                }
                return vectorBuf;
            } else if (fieldData instanceof InternalArray) {
                InternalArray array = (InternalArray) fieldData;
                checkDimension(array.size());
                for (int i = 0; i < dim; i++) {
                    if (array.isNullAt(i)) {
                        throw new IllegalArgumentException(
                                "Vector element at index " + i + " is null");
                    }
                    float v = array.getFloat(i);
                    checkFinite(v, absoluteRowId, i);
                    vectorBuf[i] = v;
                }
                return vectorBuf;
            }
            throw new RuntimeException(
                    "Unsupported vector type: " + fieldData.getClass().getName());
        }

        private void checkDimension(int actualDim) {
            if (actualDim != dim) {
                throw new IllegalArgumentException(
                        String.format(
                                "Vector dimension mismatch: expected %d, but got %d",
                                dim, actualDim));
            }
        }

        private void checkFinite(float value, long absoluteRowId, int elementIndex) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        String.format(
                                "Vector element at rowId=%d, index=%d is %s",
                                absoluteRowId, elementIndex, Float.toString(value)));
            }
        }

        private void ensureNotFinished() {
            if (finished) {
                throw new IllegalStateException(
                        "Native vector global model training has finished.");
            }
        }

        private void closeTrainerQuietly() {
            trainer.close();
        }
    }

    private static int parseDimension(Map<String, String> options) {
        String dimension = options.get("dimension");
        if (dimension == null) {
            throw new IllegalArgumentException(
                    "Native vector training requires option 'dimension'.");
        }
        int dim = Integer.parseInt(dimension);
        if (dim <= 0) {
            throw new IllegalArgumentException(
                    "Native vector training requires positive dimension.");
        }
        return dim;
    }

    static class NativeVectorTrainingModel implements VectorTrainingModel {

        private String indexType;
        private Map<String, String> options;
        private transient VectorIndexTraining training;
        private transient byte[] serializedTraining;
        private transient String modelDigest;

        private NativeVectorTrainingModel(
                String indexType, Map<String, String> options, VectorIndexTraining training) {
            this.indexType = indexType;
            this.options = new LinkedHashMap<>(options);
            this.training = training;
        }

        static NativeVectorTrainingModel load(
                String indexType, Map<String, String> options, byte[] payload) {
            NativeVectorTrainingModel model =
                    new NativeVectorTrainingModel(
                            indexType, options, VectorIndexTraining.deserialize(payload));
            model.serializedTraining = payload;
            return model;
        }

        @Override
        public VectorCentroidModel centroids() {
            return new NativeVectorCentroidModel(training());
        }

        @Override
        public String modelDigest() throws IOException {
            if (modelDigest == null) {
                modelDigest = VectorModelDigest.sha256(serializedTraining());
            }
            return modelDigest;
        }

        @Override
        public GlobalIndexSingleColumnWriter createCentroidShardIndexWriter(
                GlobalIndexFileWriter fileWriter, int centroid, Map<String, String> options) {
            return new NativeCentroidShardIndexWriter(fileWriter, this, centroid, options);
        }

        @Override
        public void serializeNativeModelPayloadTo(OutputStream out) throws IOException {
            out.write(serializedTraining());
        }

        VectorIndexWriter createIvfPqCentroidShardWriter(int centroid) {
            return new VectorIndexWriter(training(), centroid);
        }

        void addIvfPqCentroidVectors(
                VectorIndexWriter writer, long[] ids, float[] vectors, int vectorCount) {
            writer.addIvfPqCentroidVectors(ids, vectors, vectorCount);
        }

        @Override
        public void close() throws IOException {
            if (training != null) {
                training.close();
                training = null;
            }
        }

        private byte[] serializedTraining() throws IOException {
            if (serializedTraining == null) {
                serializedTraining = training().serialize();
            }
            return serializedTraining;
        }

        private VectorIndexTraining training() {
            if (training == null) {
                throw new IllegalStateException("Native vector training model has been closed.");
            }
            return training;
        }
    }

    private static class NativeVectorCentroidModel implements VectorCentroidModel {

        private final VectorIndexTraining training;

        private NativeVectorCentroidModel(VectorIndexTraining training) {
            this.training = training;
        }

        @Override
        public int nlist() {
            int nlist = training.nlist();
            if (nlist <= 0) {
                throw new IllegalStateException(
                        "Native vector training returned non-positive nlist: " + nlist);
            }
            return nlist;
        }

        @Override
        public int assignCentroid(float[] vector) {
            return training.assignCentroid(vector);
        }

        @Override
        public int[] nearestCentroids(float[] queryVector, int nprobe) {
            return training.findNearestCentroids(queryVector, nprobe);
        }
    }
}
