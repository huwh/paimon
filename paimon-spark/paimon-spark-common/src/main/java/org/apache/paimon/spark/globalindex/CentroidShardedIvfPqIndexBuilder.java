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

import org.apache.paimon.data.InternalArray;
import org.apache.paimon.data.InternalRow;
import org.apache.paimon.globalindex.GlobalIndexFileReadWrite;
import org.apache.paimon.globalindex.GlobalIndexSingleColumnWriter;
import org.apache.paimon.globalindex.IndexFileKind;
import org.apache.paimon.globalindex.IvfPqShard;
import org.apache.paimon.globalindex.ResultEntry;
import org.apache.paimon.table.FileStoreTable;
import org.apache.paimon.table.SpecialFields;
import org.apache.paimon.types.DataField;
import org.apache.paimon.types.RowType;
import org.apache.paimon.types.VectorType;
import org.apache.paimon.utils.CloseableIterator;
import org.apache.paimon.utils.Range;
import org.apache.paimon.vector.index.VectorCentroidModel;
import org.apache.paimon.vector.index.VectorGlobalIndexFileMeta;
import org.apache.paimon.vector.index.VectorGlobalModelTrainer;
import org.apache.paimon.vector.index.VectorGlobalModelTrainers;
import org.apache.paimon.vector.index.VectorIndexMeta;
import org.apache.paimon.vector.index.VectorTrainingModel;

import javax.annotation.Nullable;

import java.io.IOException;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import scala.Tuple2;

/** Builder for centroid-sharded IVF/PQ global index. */
public class CentroidShardedIvfPqIndexBuilder implements Serializable, AutoCloseable {

    private static final long serialVersionUID = 1L;

    private final FileStoreTable table;
    private final RowType readType;
    private final DataField indexField;
    private final List<Range> rowRanges;
    @Nullable private transient VectorTrainingModel trainingModel;
    @Nullable private final String trainingModelBackend;
    @Nullable private final String trainingModelIndexType;
    @Nullable private byte[] trainingModelPayload;
    private final Map<String, String> nativeOptions;

    /** Creates a builder used by scan-side tasks to collect centroid training samples. */
    public static CentroidShardedIvfPqIndexBuilder forTrainingSamples(
            FileStoreTable table, RowType readType, DataField indexField, List<Range> rowRanges) {
        return new CentroidShardedIvfPqIndexBuilder(
                table,
                readType,
                indexField,
                rowRanges,
                null,
                null,
                null,
                null,
                Collections.emptyMap());
    }

    /** Creates a builder used by scan-side tasks to add vectors into centroid buckets. */
    public static CentroidShardedIvfPqIndexBuilder forCentroidAssignment(
            FileStoreTable table,
            RowType readType,
            DataField indexField,
            List<Range> rowRanges,
            String backend,
            String indexType,
            Map<String, String> nativeOptions,
            byte[] trainingModelPayload) {
        return new CentroidShardedIvfPqIndexBuilder(
                table,
                readType,
                indexField,
                rowRanges,
                null,
                backend,
                indexType,
                trainingModelPayload,
                nativeOptions);
    }

    /**
     * Creates an assignment builder whose model payload will be supplied from a Spark broadcast on
     * the executor.
     */
    public static CentroidShardedIvfPqIndexBuilder forCentroidAssignment(
            FileStoreTable table,
            RowType readType,
            DataField indexField,
            List<Range> rowRanges,
            String backend,
            String indexType,
            Map<String, String> nativeOptions) {
        return new CentroidShardedIvfPqIndexBuilder(
                table,
                readType,
                indexField,
                rowRanges,
                null,
                backend,
                indexType,
                null,
                nativeOptions);
    }

    /** Creates a builder used by shuffle-side tasks to write one or more centroid shard files. */
    public static CentroidShardedIvfPqIndexBuilder forShardBuild(
            FileStoreTable table,
            String backend,
            String indexType,
            Map<String, String> nativeOptions,
            byte[] trainingModelPayload) {
        return new CentroidShardedIvfPqIndexBuilder(
                table,
                null,
                null,
                Collections.emptyList(),
                null,
                backend,
                indexType,
                trainingModelPayload,
                nativeOptions);
    }

    /** Creates a shard builder whose model payload is supplied from a Spark broadcast. */
    public static CentroidShardedIvfPqIndexBuilder forShardBuild(
            FileStoreTable table,
            String backend,
            String indexType,
            Map<String, String> nativeOptions) {
        return new CentroidShardedIvfPqIndexBuilder(
                table,
                null,
                null,
                Collections.emptyList(),
                null,
                backend,
                indexType,
                null,
                nativeOptions);
    }

    /** Creates a builder used by driver to write the partition-level global index file. */
    public static CentroidShardedIvfPqIndexBuilder forGlobalIndexFile(
            FileStoreTable table,
            VectorTrainingModel trainingModel,
            Map<String, String> nativeOptions) {
        return new CentroidShardedIvfPqIndexBuilder(
                table,
                null,
                null,
                Collections.emptyList(),
                trainingModel,
                null,
                null,
                null,
                nativeOptions);
    }

    private CentroidShardedIvfPqIndexBuilder(
            FileStoreTable table,
            @Nullable RowType readType,
            @Nullable DataField indexField,
            List<Range> rowRanges,
            @Nullable VectorTrainingModel trainingModel,
            @Nullable String trainingModelBackend,
            @Nullable String trainingModelIndexType,
            @Nullable byte[] trainingModelPayload,
            Map<String, String> nativeOptions) {
        this.table = table;
        this.readType = readType;
        this.indexField = indexField;
        this.rowRanges = new ArrayList<>(rowRanges);
        this.trainingModel = trainingModel;
        this.trainingModelBackend = trainingModelBackend;
        this.trainingModelIndexType = trainingModelIndexType;
        this.trainingModelPayload =
                trainingModelPayload == null ? null : trainingModelPayload.clone();
        this.nativeOptions = new LinkedHashMap<>(nativeOptions);
    }

    public FileStoreTable table() {
        return table;
    }

    public RowType readType() {
        return readType;
    }

    public List<TrainingSample> collectTrainingSamples(CloseableIterator<InternalRow> rows) {
        List<TrainingSample> samples = new ArrayList<>();
        InternalRow.FieldGetter vectorGetter = vectorGetter();
        int rowIdIndex = rowIdIndex();
        while (rows.hasNext()) {
            InternalRow row = rows.next();
            long absoluteRowId = row.getLong(rowIdIndex);
            if (!containsRowId(absoluteRowId)) {
                continue;
            }
            Object vectorObject = vectorGetter.getFieldOrNull(row);
            if (vectorObject == null) {
                continue;
            }
            samples.add(new TrainingSample(absoluteRowId, toFloatVector(vectorObject)));
        }
        return samples;
    }

    /** Streams at most {@code maxSamples} eligible rows directly into the trainer. */
    public long writeTrainingSamples(
            CloseableIterator<InternalRow> rows, VectorGlobalModelTrainer trainer, long maxSamples)
            throws IOException {
        InternalRow.FieldGetter vectorGetter = vectorGetter();
        int rowIdIndex = rowIdIndex();
        long written = 0;
        while (written < maxSamples && rows.hasNext()) {
            InternalRow row = rows.next();
            long absoluteRowId = row.getLong(rowIdIndex);
            if (!containsRowId(absoluteRowId)) {
                continue;
            }
            Object vectorObject = vectorGetter.getFieldOrNull(row);
            if (vectorObject == null) {
                continue;
            }
            trainer.write(toFloatVector(vectorObject), absoluteRowId);
            written++;
        }
        return written;
    }

    /** Installs the immutable model payload obtained from a Spark broadcast. */
    public void setBroadcastTrainingModelPayload(byte[] trainingModelPayload) {
        if (trainingModel != null) {
            throw new IllegalStateException("The vector training model is already initialized.");
        }
        this.trainingModelPayload = requireNonNull(trainingModelPayload, "trainingModelPayload");
    }

    public List<Tuple2<Integer, AssignedVector>> assignCentroidVectors(
            CloseableIterator<InternalRow> rows) {
        List<Tuple2<Integer, AssignedVector>> result = new ArrayList<>();
        InternalRow.FieldGetter vectorGetter = vectorGetter();
        int rowIdIndex = rowIdIndex();
        while (rows.hasNext()) {
            InternalRow row = rows.next();
            long absoluteRowId = row.getLong(rowIdIndex);
            if (!containsRowId(absoluteRowId)) {
                continue;
            }
            Object vectorObject = vectorGetter.getFieldOrNull(row);
            if (vectorObject == null) {
                continue;
            }
            float[] vector = toFloatVector(vectorObject);
            int centroid = centroids().assignCentroid(vector);
            result.add(new Tuple2<>(centroid, new AssignedVector(absoluteRowId, vector)));
        }
        return result;
    }

    public Iterator<Tuple2<Integer, AssignedVector>> assignCentroidVectorsLazy(
            CloseableIterator<InternalRow> rows) {
        return new Iterator<Tuple2<Integer, AssignedVector>>() {

            private final InternalRow.FieldGetter vectorGetter = vectorGetter();
            private final int rowIdIndex = rowIdIndex();
            private Tuple2<Integer, AssignedVector> next;
            private boolean nextReady;

            @Override
            public boolean hasNext() {
                prepareNext();
                return nextReady;
            }

            @Override
            public Tuple2<Integer, AssignedVector> next() {
                prepareNext();
                if (!nextReady) {
                    throw new NoSuchElementException();
                }
                Tuple2<Integer, AssignedVector> result = next;
                next = null;
                nextReady = false;
                return result;
            }

            private void prepareNext() {
                if (nextReady) {
                    return;
                }
                while (rows.hasNext()) {
                    InternalRow row = rows.next();
                    long absoluteRowId = row.getLong(rowIdIndex);
                    if (!containsRowId(absoluteRowId)) {
                        continue;
                    }
                    Object vectorObject = vectorGetter.getFieldOrNull(row);
                    if (vectorObject == null) {
                        continue;
                    }
                    float[] vector = toFloatVector(vectorObject);
                    int centroid = centroids().assignCentroid(vector);
                    next = new Tuple2<>(centroid, new AssignedVector(absoluteRowId, vector));
                    nextReady = true;
                    return;
                }
            }
        };
    }

    /** Builds centroid shards from input sorted by centroid id. */
    public List<ShardBuildResult> buildCentroidShards(
            Iterator<Tuple2<Integer, AssignedVector>> assignedVectors) throws IOException {
        GlobalIndexFileReadWrite fileReadWrite =
                new GlobalIndexFileReadWrite(
                        table.fileIO(), table.store().pathFactory().globalIndexFileFactory());
        List<ShardBuildResult> results = new ArrayList<>();
        int currentCentroid = -1;
        GlobalIndexSingleColumnWriter writer = null;
        try {
            while (assignedVectors.hasNext()) {
                Tuple2<Integer, AssignedVector> tuple = assignedVectors.next();
                int centroid = tuple._1();
                AssignedVector vector = tuple._2();
                if (writer != null && centroid < currentCentroid) {
                    throw new IllegalArgumentException(
                            "Centroid shard input must be sorted by centroid id.");
                }
                if (writer == null || centroid != currentCentroid) {
                    if (writer != null) {
                        finishCentroidWriter(results, currentCentroid, writer);
                        writer = null;
                    }
                    // The physical native index build is delegated to the shard writer. For the
                    // native backend, NativeCentroidShardIndexWriter derives a centroid-specific
                    // VectorIndexTraining from the global model, constructs VectorIndexWriter,
                    // feeds vectors through addVectors in batches, and materializes the index file
                    // in finish().
                    writer =
                            trainingModel()
                                    .createCentroidShardIndexWriter(
                                            fileReadWrite, centroid, nativeOptions);
                    currentCentroid = centroid;
                }
                writer.write(vector.vector(), vector.absoluteRowId());
            }
            if (writer != null) {
                finishCentroidWriter(results, currentCentroid, writer);
                writer = null;
            }
            return results;
        } finally {
            if (writer != null) {
                closeWriter(writer);
            }
        }
    }

    private static void finishCentroidWriter(
            List<ShardBuildResult> results,
            int centroid,
            GlobalIndexSingleColumnWriter writer)
            throws IOException {
        try {
            for (ResultEntry resultEntry : writer.finish()) {
                results.add(new ShardBuildResult(centroid, resultEntry));
            }
        } finally {
            closeWriter(writer);
        }
    }

    public ResultEntry writeVectorGlobalIndexFile(
            String fileName,
            String indexType,
            int indexFieldId,
            @Nullable int[] extraFieldIds,
            Range rowRange)
            throws IOException {
        GlobalIndexFileReadWrite fileReadWrite =
                new GlobalIndexFileReadWrite(
                        table.fileIO(), table.store().pathFactory().globalIndexFileFactory());
        try (OutputStream out = fileReadWrite.newOutputStream(fileName)) {
            VectorGlobalIndexFileMeta.centroidSharded().writeWithPayload(out, trainingModel());
        }
        return new ResultEntry(
                fileName,
                0,
                VectorIndexMeta.routingModel(
                                IvfPqShard.CENTROID_BASED,
                                trainingModel().centroids().nlist(),
                                trainingModel().modelDigest())
                        .serialize(),
                IndexFileKind.ROUTING_MODEL);
    }

    public static Map<String, String> nativeOptions(
            String indexType, DataField indexField, org.apache.paimon.options.Options options) {
        Map<String, String> nativeOptions = new LinkedHashMap<>();
        Map<String, String> optionMap = options.toMap();
        collectNativeOptions(nativeOptions, optionMap, indexType + ".");
        collectNativeOptions(nativeOptions, optionMap, "fields." + indexField.name() + ".");
        nativeOptions.put("index.type", indexType.replace('-', '_'));
        nativeOptions.putIfAbsent("metric", "inner_product");
        if (indexField.type() instanceof VectorType) {
            nativeOptions.put(
                    "dimension", String.valueOf(((VectorType) indexField.type()).getLength()));
        }
        return nativeOptions;
    }

    public static Range coveringRange(List<Range> ranges) {
        long from = Long.MAX_VALUE;
        long to = Long.MIN_VALUE;
        for (Range range : ranges) {
            from = Math.min(from, range.from);
            to = Math.max(to, range.to);
        }
        return new Range(from, to);
    }

    private static void collectNativeOptions(
            Map<String, String> nativeOptions, Map<String, String> optionMap, String prefix) {
        for (Map.Entry<String, String> entry : optionMap.entrySet()) {
            if (!entry.getKey().startsWith(prefix)) {
                continue;
            }
            String nativeKey = nativeOptionKey(entry.getKey().substring(prefix.length()));
            if (nativeKey != null) {
                nativeOptions.put(nativeKey, entry.getValue());
            }
        }
    }

    private static String nativeOptionKey(String key) {
        if ("dimension".equals(key) || "index.dimension".equals(key)) {
            return "dimension";
        }
        if ("metric".equals(key) || "distance.metric".equals(key)) {
            return "metric";
        }
        if ("nlist".equals(key)
                || "expected-vector-count".equals(key)
                || "pq.m".equals(key)
                || "pq.code-ratio".equals(key)
                || "pq.bits".equals(key)
                || "rq.bits".equals(key)
                || "target-recall".equals(key)
                || "max-bytes-per-vector".equals(key)
                || "deployment-profile".equals(key)) {
            return key;
        }
        if ("pq.use-opq".equals(key) || "use-opq".equals(key)) {
            return "use-opq";
        }
        return null;
    }

    private boolean containsRowId(long rowId) {
        for (Range range : rowRanges) {
            if (rowId >= range.from && rowId <= range.to) {
                return true;
            }
        }
        return false;
    }

    private InternalRow.FieldGetter vectorGetter() {
        return InternalRow.createFieldGetter(
                indexField().type(), readType().getFieldIndex(indexField().name()));
    }

    private int rowIdIndex() {
        return readType().getFieldIndex(SpecialFields.ROW_ID.name());
    }

    private VectorCentroidModel centroids() {
        return trainingModel().centroids();
    }

    private VectorTrainingModel trainingModel() {
        if (trainingModel == null) {
            try {
                trainingModel =
                        VectorGlobalModelTrainers.load(
                                requireNonNull(trainingModelBackend, "trainingModelBackend"),
                                requireNonNull(trainingModelIndexType, "trainingModelIndexType"),
                                nativeOptions,
                                requireNonNull(trainingModelPayload, "trainingModelPayload"));
            } catch (IOException e) {
                throw new IllegalStateException("Failed to restore vector training model.", e);
            }
        }
        return trainingModel;
    }

    private DataField indexField() {
        return requireNonNull(indexField, "indexField");
    }

    @Override
    public void close() throws IOException {
        if (trainingModel != null) {
            trainingModel.close();
            trainingModel = null;
        }
    }

    private static <T> T requireNonNull(@Nullable T value, String name) {
        if (value == null) {
            throw new IllegalStateException(name + " is not initialized.");
        }
        return value;
    }

    private static void closeWriter(GlobalIndexSingleColumnWriter writer) throws IOException {
        if (writer instanceof AutoCloseable) {
            try {
                ((AutoCloseable) writer).close();
            } catch (IOException e) {
                throw e;
            } catch (Exception e) {
                throw new IOException("Failed to close centroid shard writer.", e);
            }
        }
    }

    private static float[] toFloatVector(Object vectorObject) {
        if (vectorObject instanceof InternalArray) {
            return ((InternalArray) vectorObject).toFloatArray();
        }
        if (vectorObject instanceof float[]) {
            return ((float[]) vectorObject).clone();
        }
        throw new IllegalArgumentException(
                "Unsupported vector value class: " + vectorObject.getClass().getName());
    }

    /** One vector selected for centroid training. */
    public static class TrainingSample implements Serializable {

        private static final long serialVersionUID = 1L;

        private final long absoluteRowId;
        private final float[] vector;

        public TrainingSample(long absoluteRowId, float[] vector) {
            this.absoluteRowId = absoluteRowId;
            this.vector = vector.clone();
        }

        public long absoluteRowId() {
            return absoluteRowId;
        }

        public float[] vector() {
            return vector.clone();
        }
    }

    /** One vector assigned to a centroid shard. */
    public static class AssignedVector implements Serializable {

        private static final long serialVersionUID = 1L;

        private final long absoluteRowId;
        private final float[] vector;

        public AssignedVector(long absoluteRowId, float[] vector) {
            this.absoluteRowId = absoluteRowId;
            this.vector = vector.clone();
        }

        public long absoluteRowId() {
            return absoluteRowId;
        }

        public float[] vector() {
            return vector.clone();
        }
    }

    /** Result of building one centroid shard file. */
    public static class ShardBuildResult implements Serializable {

        private static final long serialVersionUID = 1L;

        private final int centroid;
        private final String fileName;
        private final long rowCount;
        @Nullable private final byte[] meta;
        @Nullable private final Range rowRange;

        public ShardBuildResult(int centroid, ResultEntry resultEntry) {
            this.centroid = centroid;
            this.fileName = resultEntry.fileName();
            this.rowCount = resultEntry.rowCount();
            this.meta = resultEntry.meta() == null ? null : resultEntry.meta().clone();
            this.rowRange = resultEntry.rowRange();
        }

        public int centroid() {
            return centroid;
        }

        public String fileName() {
            return fileName;
        }

        public long rowCount() {
            return rowCount;
        }

        @Nullable
        public byte[] meta() {
            return meta == null ? null : meta.clone();
        }

        public ResultEntry toResultEntry() {
            return new ResultEntry(
                    fileName, rowCount, meta == null ? null : meta.clone(), rowRange);
        }
    }
}
