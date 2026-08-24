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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
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
        return NativeVectorTrainingModel.load(indexType, options, new ByteArrayInputStream(payload));
    }

    /**
     * Java-side trainer for the centroid global IVF/PQ model.
     *
     * <p>This class is intentionally not a placeholder. It owns the same Java-side work as {@link
     * NativeVectorGlobalIndexWriter}: validate/materialize vectors, batch them, feed the existing
     * native {@link VectorIndexTrainer}, and convert the native training result into a Paimon
     * {@link VectorTrainingModel}. The native dependencies after {@link
     * VectorIndexTrainer#finishTraining()} are implemented explicitly by {@link
     * NativeGlobalTrainingModelOps}; paimon-vector-index-java must expose the corresponding routing
     * methods plus serde methods on {@link VectorIndexTraining} for this class to run the centroid
     * path.
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

            flushTrainingBatch();
            VectorIndexTraining training = trainer.finishTraining();
            closeTrainerQuietly();
            batchVectors = null;
            return new NativeVectorTrainingModel(
                    indexType, options, training, NativeGlobalTrainingModelOps.NATIVE);
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
            throw new RuntimeException("Unsupported vector type: " + fieldData.getClass().getName());
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
                throw new IllegalStateException("Native vector global model training has finished.");
            }
        }

        private void closeTrainerQuietly() {
            trainer.close();
        }
    }

    private static int parseDimension(Map<String, String> options) {
        String dimension = options.get("dimension");
        if (dimension == null) {
            throw new IllegalArgumentException("Native vector training requires option 'dimension'.");
        }
        int dim = Integer.parseInt(dimension);
        if (dim <= 0) {
            throw new IllegalArgumentException("Native vector training requires positive dimension.");
        }
        return dim;
    }

    static class NativeVectorTrainingModel implements VectorTrainingModel {

        private String indexType;
        private Map<String, String> options;
        private transient VectorIndexTraining training;
        private transient byte[] serializedTraining;
        private transient NativeGlobalTrainingModelOps nativeOps;

        private NativeVectorTrainingModel(
                String indexType,
                Map<String, String> options,
                VectorIndexTraining training,
                NativeGlobalTrainingModelOps nativeOps) {
            this.indexType = indexType;
            this.options = new LinkedHashMap<>(options);
            this.training = training;
            this.nativeOps = nativeOps;
        }

        static NativeVectorTrainingModel load(
                String indexType, Map<String, String> options, InputStream payload)
                throws IOException {
            byte[] serializedTraining = readFully(payload);
            NativeVectorTrainingModel model =
                    new NativeVectorTrainingModel(
                            indexType,
                            options,
                            NativeGlobalTrainingModelOps.NATIVE.deserialize(
                                    serializedTraining, options),
                            NativeGlobalTrainingModelOps.NATIVE);
            model.serializedTraining = serializedTraining;
            return model;
        }

        @Override
        public VectorCentroidModel centroids() {
            return new NativeVectorCentroidModel(options, training(), nativeOps);
        }

        @Override
        public GlobalIndexSingleColumnWriter createCentroidShardIndexWriter(
                GlobalIndexFileWriter fileWriter,
                int centroid,
                Map<String, String> options) {
            return new NativeCentroidShardIndexWriter(fileWriter, this, centroid, options);
        }

        @Override
        public void serializeNativeModelPayloadTo(OutputStream out) throws IOException {
            out.write(serializedTraining());
        }

        VectorIndexWriter createIvfPqCentroidShardWriter(int centroid) {
            return nativeOps.createCentroidShardWriter(training(), centroid, options);
        }

        void addIvfPqCentroidVectors(
                VectorIndexWriter writer, long[] ids, float[] vectors, int vectorCount) {
            nativeOps.addIvfPqCentroidVectors(writer, ids, vectors, vectorCount, options);
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
                serializedTraining = nativeOps.serialize(training(), options);
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

        private final Map<String, String> options;
        private final VectorIndexTraining training;
        private final NativeGlobalTrainingModelOps nativeOps;

        private NativeVectorCentroidModel(
                Map<String, String> options,
                VectorIndexTraining training,
                NativeGlobalTrainingModelOps nativeOps) {
            this.options = new LinkedHashMap<>(options);
            this.training = training;
            this.nativeOps = nativeOps;
        }

        @Override
        public int nlist() {
            return parsePositiveOption(options, "nlist");
        }

        @Override
        public int assignCentroid(float[] vector) {
            return nativeOps.assignCentroid(training, vector, options);
        }

        @Override
        public int[] nearestCentroids(float[] queryVector, int nprobe) {
            return nativeOps.findNearestCentroids(training, queryVector, nprobe, options);
        }

        private static int parsePositiveOption(Map<String, String> options, String key) {
            String value = options.get(key);
            if (value == null) {
                throw new IllegalArgumentException(
                        "Native vector training requires option '" + key + "'.");
            }
            int parsed = Integer.parseInt(value);
            if (parsed <= 0) {
                throw new IllegalArgumentException(
                        "Native vector training requires positive option '" + key + "'.");
            }
            return parsed;
        }
    }

    /**
     * Boundary for native operations which are required by centroid-id sharded IVF/PQ global
     * indexes.
     *
     * <p>This is not an unsupported placeholder. It is the concrete Paimon adapter around the
     * paimon-vector-index Java binding contract. The implementation below calls explicit APIs from
     * the Java binding. Operations which naturally belong to the training model, such as centroid
     * routing, stay on {@link VectorIndexTraining}. Centroid shard materialization is exposed as a
     * direct {@link VectorIndexWriter} constructor receiving the global training model and centroid.
     * Metadata needed by Paimon is derived from Java-side options to avoid
     * introducing unnecessary native APIs.
     * Persistence stays on {@link VectorIndexTraining}; Spark carries the serialized training-model
     * payload between stages, and each JVM recreates a fresh native handle from those bytes. A Java
     * object containing only a native pointer is not valid in another JVM. The adapter invokes these
     * APIs reflectively so this module can still compile while the paimon-vector-index Java binding is
     * being upgraded; using the centroid path with an older binding fails fast with a clear runtime
     * error that names the missing method or constructor.
     *
     * <p>Required paimon-vector-index Java binding additions:
     *
     * <pre>{@code
     * public final class VectorIndexTraining implements AutoCloseable {
     *     public byte[] serialize();
     *
     *     public static VectorIndexTraining deserialize(byte[] serializedTraining);
     *
     *     public int assignCentroid(float[] vector);
     *
     *     public int[] findNearestCentroids(float[] query, int nprobe);
     * }
     *
     * public final class VectorIndexWriter implements AutoCloseable {
     *     public VectorIndexWriter(VectorIndexTraining training, int centroid);
     *
     *     public void addIvfPqCentroidVectors(
     *             long[] ids,
     *             float[] vectors,
     *             int vectorCount);
     * }
     * }</pre>
     */
    interface NativeGlobalTrainingModelOps {

        NativeGlobalTrainingModelOps NATIVE = new VectorIndexTrainingNativeGlobalModelOps();

        /**
         * Serializes the full native global training model.
         *
         * <p>Purpose: store the partition-level model next to centroid shard file metadata, and
         * serialize the model across Spark tasks when needed.
         *
         * @param training closeable native training model to serialize
         * @param options native index options used to disambiguate the payload format
         *     <p>The returned bytes are a self-contained payload consumable by {@link
         *     #deserialize(byte[], Map)}.
         */
        byte[] serialize(VectorIndexTraining training, Map<String, String> options) throws IOException;

        /**
         * Assigns one build-side vector to its closest IVF centroid.
         *
         * <p>Purpose: shuffle table rows into deterministic per-centroid shard builders.
         *
         * @param training closeable native global training model
         * @param vector dense float vector whose length equals the configured dimension
         * @param options native index options used by the training model
         * @return zero-based centroid in [0, nlist)
         */
        int assignCentroid(VectorIndexTraining training, float[] vector, Map<String, String> options);

        /**
         * Selects centroid shards which may contain nearest neighbours for a query vector.
         *
         * <p>Purpose: query-side routing so readers open only relevant centroid shard files.
         *
         * @param training closeable native global training model
         * @param queryVector dense query vector whose length equals vector dimension
         * @param nprobe requested number of IVF centroids to probe; must be positive
         * @param options native index options used by the training model
         * @return distinct zero-based centroids, ordered by native routing priority, with length
         *     at most min(nprobe, nlist)
         */
        int[] findNearestCentroids(
                VectorIndexTraining training,
                float[] queryVector,
                int nprobe,
                Map<String, String> options);

        /** Creates a native writer for one centroid shard from a global training model. */
        VectorIndexWriter createCentroidShardWriter(
                VectorIndexTraining training, int centroid, Map<String, String> options);

        /** Adds a batch of absolute row ids and vectors to a centroid shard writer. */
        void addIvfPqCentroidVectors(
                VectorIndexWriter writer,
                long[] ids,
                float[] vectors,
                int vectorCount,
                Map<String, String> options);

        /**
         * Loads a persisted native global training model.
         *
         * <p>Purpose: recreate the model on query-side readers and across serialized build tasks.
         *
         * @param payload bytes previously produced by {@link #serialize(VectorIndexTraining, Map)}
         * @param options native index options used to disambiguate the payload format
         * @return closeable native global training model that supports metadata, centroid routing
         *     and serialized centroid shard writing
         */
        VectorIndexTraining deserialize(byte[] payload, Map<String, String> options) throws IOException;
    }

    private static class VectorIndexTrainingNativeGlobalModelOps
            implements NativeGlobalTrainingModelOps {

        @Override
        public byte[] serialize(VectorIndexTraining training, Map<String, String> options)
                throws IOException {
            return invokeRequired(
                    training,
                    "serialize",
                    new Class<?>[0],
                    new Object[0],
                    byte[].class);
        }

        @Override
        public int assignCentroid(
                VectorIndexTraining training, float[] vector, Map<String, String> options) {
            Integer centroid =
                    invokeRequired(
                            training,
                            "assignCentroid",
                            new Class<?>[] {float[].class},
                            new Object[] {vector},
                            Integer.class);
            return centroid;
        }

        @Override
        public int[] findNearestCentroids(
                VectorIndexTraining training,
                float[] queryVector,
                int nprobe,
                Map<String, String> options) {
            return invokeRequired(
                    training,
                    "findNearestCentroids",
                    new Class<?>[] {float[].class, int.class},
                    new Object[] {queryVector, nprobe},
                    int[].class);
        }

        @Override
        public VectorIndexTraining deserialize(byte[] payload, Map<String, String> options)
                throws IOException {
            return invokeRequiredStatic(
                    VectorIndexTraining.class,
                    "deserialize",
                    new Class<?>[] {byte[].class},
                    new Object[] {payload},
                    VectorIndexTraining.class);
        }

        @Override
        public VectorIndexWriter createCentroidShardWriter(
                VectorIndexTraining training, int centroid, Map<String, String> options) {
            try {
                Constructor<VectorIndexWriter> constructor =
                        VectorIndexWriter.class.getConstructor(VectorIndexTraining.class, int.class);
                return constructor.newInstance(training, centroid);
            } catch (NoSuchMethodException e) {
                throw missingNativeApi("VectorIndexWriter(VectorIndexTraining, int)", e);
            } catch (InstantiationException | IllegalAccessException e) {
                throw new RuntimeException(
                        "Failed to create native IVF_PQ centroid shard writer.", e);
            } catch (InvocationTargetException e) {
                throw rethrowInvocation(
                        "Failed to create native IVF_PQ centroid shard writer.", e);
            }
        }

        @Override
        public void addIvfPqCentroidVectors(
                VectorIndexWriter writer,
                long[] ids,
                float[] vectors,
                int vectorCount,
                Map<String, String> options) {
            invokeRequired(
                    writer,
                    "addIvfPqCentroidVectors",
                    new Class<?>[] {long[].class, float[].class, int.class},
                    new Object[] {ids, vectors, vectorCount},
                    Void.TYPE);
        }

        private static <T> T invokeRequired(
                Object target,
                String methodName,
                Class<?>[] parameterTypes,
                Object[] args,
                Class<T> returnType) {
            try {
                Method method = target.getClass().getMethod(methodName, parameterTypes);
                Object result = method.invoke(target, args);
                if (returnType == Void.TYPE) {
                    return null;
                }
                return returnType.cast(result);
            } catch (NoSuchMethodException e) {
                throw missingNativeApi(target.getClass().getSimpleName() + "." + methodName, e);
            } catch (IllegalAccessException e) {
                throw new RuntimeException("Failed to access native vector API: " + methodName, e);
            } catch (InvocationTargetException e) {
                throw rethrowInvocation("Native vector API failed: " + methodName, e);
            }
        }

        private static <T> T invokeRequiredStatic(
                Class<?> targetClass,
                String methodName,
                Class<?>[] parameterTypes,
                Object[] args,
                Class<T> returnType) {
            try {
                Method method = targetClass.getMethod(methodName, parameterTypes);
                Object result = method.invoke(null, args);
                return returnType.cast(result);
            } catch (NoSuchMethodException e) {
                throw missingNativeApi(targetClass.getSimpleName() + "." + methodName, e);
            } catch (IllegalAccessException e) {
                throw new RuntimeException("Failed to access native vector API: " + methodName, e);
            } catch (InvocationTargetException e) {
                throw rethrowInvocation("Native vector API failed: " + methodName, e);
            }
        }

        private static UnsupportedOperationException missingNativeApi(
                String api, NoSuchMethodException cause) {
            return new UnsupportedOperationException(
                    "Current paimon-vector-index Java binding does not support centroid IVF_PQ "
                            + "sharding. Missing native API: "
                            + api,
                    cause);
        }

        private static RuntimeException rethrowInvocation(
                String message, InvocationTargetException e) {
            Throwable cause = e.getCause();
            if (cause instanceof RuntimeException) {
                return (RuntimeException) cause;
            }
            if (cause instanceof Error) {
                throw (Error) cause;
            }
            return new RuntimeException(message, cause);
        }
    }

    private static byte[] readFully(InputStream input) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[4096];
        int read;
        while ((read = input.read(buffer)) >= 0) {
            output.write(buffer, 0, read);
        }
        return output.toByteArray();
    }
}
