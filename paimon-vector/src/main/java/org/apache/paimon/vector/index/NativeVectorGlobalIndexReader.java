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

import org.apache.paimon.fs.FileRange;
import org.apache.paimon.fs.SeekableInputStream;
import org.apache.paimon.fs.VectoredReadUtils;
import org.apache.paimon.fs.VectoredReadable;
import org.apache.paimon.globalindex.GlobalIndexIOMeta;
import org.apache.paimon.globalindex.GlobalIndexReader;
import org.apache.paimon.globalindex.GlobalIndexResult;
import org.apache.paimon.globalindex.ScoredGlobalIndexResult;
import org.apache.paimon.globalindex.io.GlobalIndexFileReader;
import org.apache.paimon.index.vector.VectorIndexInput;
import org.apache.paimon.index.vector.VectorIndexMetadata;
import org.apache.paimon.index.vector.VectorIndexReader;
import org.apache.paimon.index.vector.VectorSearchBatchResult;
import org.apache.paimon.index.vector.VectorSearchParams;
import org.apache.paimon.index.vector.VectorSearchResult;
import org.apache.paimon.predicate.BatchVectorSearch;
import org.apache.paimon.predicate.FieldRef;
import org.apache.paimon.predicate.VectorSearch;
import org.apache.paimon.types.ArrayType;
import org.apache.paimon.types.DataType;
import org.apache.paimon.types.FloatType;
import org.apache.paimon.types.VectorType;
import org.apache.paimon.utils.IOUtils;
import org.apache.paimon.utils.RoaringNavigableMap64;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;

import static org.apache.paimon.utils.Preconditions.checkArgument;

/**
 * Vector global index reader using paimon-vector-index-java.
 *
 * <p>Each shard has exactly one vector index file. The reader lazily opens the index and performs
 * vector similarity search.
 */
public class NativeVectorGlobalIndexReader implements GlobalIndexReader {

    private static final String NPROBE_PARAMETER = "ivf.nprobe";
    private static final String L_SEARCH_PARAMETER = "diskann.l_search";
    private static final String MAX_INITIAL_FILTER_EXPANSION_FACTOR_PARAMETER =
            "ivf.max_initial_filter_expansion_factor";
    private static final String IVF_PQ_BATCH_TABLE_REUSE_PARAMETER = "ivf_pq.batch_table_reuse";
    private static final String IVF_PQ_BATCH_TABLE_REUSE_MAX_BYTES_PARAMETER =
            "ivf_pq.batch_table_reuse.max_bytes";
    private static final String SEARCH_CENTROID_METHOD = "searchIvfPqCentroid";
    private static final String SEARCH_CENTROID_BATCH_METHOD = "searchIvfPqCentroidBatch";
    private static final int VECTOR_INDEX_MIN_SEEK_FOR_VECTOR_READS = 16 * 1024;
    private static final int VECTOR_INDEX_PARALLELISM_FOR_VECTOR_READS = 32;

    private final GlobalIndexIOMeta ioMeta;
    private final GlobalIndexFileReader fileReader;
    private final DataType fieldType;
    private final ExecutorService executor;

    private volatile VectorIndexMetadata nativeMeta;
    private volatile VectorIndexReader vectorReader;
    private SeekableInputStream openStream;

    public NativeVectorGlobalIndexReader(
            GlobalIndexFileReader fileReader,
            List<GlobalIndexIOMeta> ioMetas,
            DataType fieldType,
            ExecutorService executor) {
        checkArgument(ioMetas.size() == 1, "Expected exactly one index file per shard");
        this.executor = executor;
        this.fileReader = fileReader;
        this.ioMeta = ioMetas.get(0);
        this.fieldType = fieldType;
    }

    @Override
    public CompletableFuture<Optional<ScoredGlobalIndexResult>> visitVectorSearch(
            VectorSearch vectorSearch) {
        return CompletableFuture.supplyAsync(
                () -> {
                    try {
                        ensureLoaded();
                        return Optional.ofNullable(search(vectorSearch));
                    } catch (IOException e) {
                        throw new RuntimeException(
                                String.format(
                                        "Failed vector index search: field=%s, limit=%d",
                                        vectorSearch.fieldName(), vectorSearch.limit()),
                                e);
                    }
                },
                executor);
    }

    @Override
    public CompletableFuture<List<Optional<ScoredGlobalIndexResult>>> visitBatchVectorSearch(
            BatchVectorSearch batchVectorSearch) {
        return CompletableFuture.supplyAsync(
                () -> {
                    try {
                        ensureLoaded();
                        return searchBatch(batchVectorSearch);
                    } catch (IOException e) {
                        throw new RuntimeException(
                                String.format(
                                        "Failed batch vector index search: field=%s, limit=%d, vectorCount=%d",
                                        batchVectorSearch.fieldName(),
                                        batchVectorSearch.limit(),
                                        batchVectorSearch.vectorCount()),
                                e);
                    }
                },
                executor);
    }

    private List<Optional<ScoredGlobalIndexResult>> searchBatch(BatchVectorSearch batchVectorSearch)
            throws IOException {
        int n = batchVectorSearch.vectorCount();
        // Single vector: reuse the scalar path; no batching benefit.
        if (n == 1) {
            List<Optional<ScoredGlobalIndexResult>> results = new ArrayList<>(1);
            results.add(Optional.ofNullable(search(batchVectorSearch.forIndex(0))));
            return results;
        }

        float[][] vectors = batchVectorSearch.vectors();
        for (float[] vector : vectors) {
            validateSearchVector(vector);
        }
        int dim = nativeMeta.dimension();
        String metric = nativeMeta.metric();

        SearchScope scope =
                resolveScope(batchVectorSearch.includeRowIds(), batchVectorSearch.limit());
        if (scope == null) {
            return emptyResults(n);
        }
        VectorSearchParams searchParams =
                batchSearchParams(batchVectorSearch.options(), scope.effectiveK);

        // Flatten query vectors into one contiguous array for a single native call.
        float[] queries = flattenVectors(vectors, n, dim);

        VectorSearchBatchResult batchResult =
                scope.filterBytes != null
                        ? vectorReader.searchBatch(queries, n, searchParams, scope.filterBytes)
                        : vectorReader.searchBatch(queries, n, searchParams);

        // result i corresponds to vectors[i], matching input order.
        List<Optional<ScoredGlobalIndexResult>> results = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            results.add(
                    buildScoredResult(
                            batchResult.idsForQuery(i), batchResult.distancesForQuery(i), metric));
        }
        return results;
    }

    List<Optional<ScoredGlobalIndexResult>> searchCentroidShardBatch(
            BatchVectorSearch batchVectorSearch,
            int centroid,
            NativeCentroidRoutedIndexReader.CentroidShardSearchMode searchMode)
            throws IOException {
        ensureLoaded();
        if (searchMode == NativeCentroidRoutedIndexReader.CentroidShardSearchMode.ALL_LISTS) {
            return searchBatch(withFullIvfProbe(batchVectorSearch));
        }

        int n = batchVectorSearch.vectorCount();
        // Single vector: reuse the scalar centroid path; no batching benefit.
        if (n == 1) {
            List<Optional<ScoredGlobalIndexResult>> results = new ArrayList<>(1);
            results.add(
                    searchCentroidShard(
                            batchVectorSearch.forIndex(0), centroid, searchMode));
            return results;
        }

        float[][] vectors = batchVectorSearch.vectors();
        for (float[] vector : vectors) {
            validateSearchVector(vector);
        }
        int dim = nativeMeta.dimension();
        String metric = nativeMeta.metric();

        SearchScope scope =
                resolveScope(batchVectorSearch.includeRowIds(), batchVectorSearch.limit());
        if (scope == null) {
            return emptyResults(n);
        }

        float[] queries = flattenVectors(vectors, n, dim);
        VectorSearchParams params = searchParams(batchVectorSearch.options(), scope.effectiveK);
        VectorSearchBatchResult batchResult =
                scope.filterBytes != null
                        ? searchIvfPqCentroidBatch(
                                queries, n, params, centroid, scope.filterBytes)
                        : searchIvfPqCentroidBatch(queries, n, params, centroid);

        List<Optional<ScoredGlobalIndexResult>> results = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            results.add(
                    buildScoredResult(
                            batchResult.idsForQuery(i), batchResult.distancesForQuery(i), metric));
        }
        return results;
    }

    private ScoredGlobalIndexResult search(VectorSearch vectorSearch) throws IOException {
        validateSearchVector(vectorSearch.vector());
        float[] queryVector = vectorSearch.vector().clone();
        int limit = vectorSearch.limit();
        String metric = nativeMeta.metric();

        SearchScope scope = resolveScope(vectorSearch.includeRowIds(), limit);
        if (scope == null) {
            return null;
        }
        VectorSearchResult result =
                scope.filterBytes != null
                        ? vectorReader.search(
                                queryVector,
                                searchParams(vectorSearch.options(), scope.effectiveK),
                                scope.filterBytes)
                        : vectorReader.search(
                                queryVector,
                                searchParams(vectorSearch.options(), scope.effectiveK));

        return buildScoredResult(result.ids(), result.distances(), metric).orElse(null);
    }

    Optional<ScoredGlobalIndexResult> searchCentroidShard(
            VectorSearch vectorSearch,
            int centroid,
            NativeCentroidRoutedIndexReader.CentroidShardSearchMode searchMode)
            throws IOException {
        ensureLoaded();
        if (searchMode == NativeCentroidRoutedIndexReader.CentroidShardSearchMode.ALL_LISTS) {
            return Optional.ofNullable(search(withFullIvfProbe(vectorSearch)));
        }

        validateSearchVector(vectorSearch.vector());
        float[] queryVector = vectorSearch.vector().clone();
        String metric = nativeMeta.metric();

        SearchScope scope = resolveScope(vectorSearch.includeRowIds(), vectorSearch.limit());
        if (scope == null) {
            return Optional.empty();
        }

        VectorSearchParams params = searchParams(vectorSearch.options(), scope.effectiveK);
        VectorSearchResult result =
                scope.filterBytes != null
                        ? searchIvfPqCentroid(queryVector, params, centroid, scope.filterBytes)
                        : searchIvfPqCentroid(queryVector, params, centroid);
        return buildScoredResult(result.ids(), result.distances(), metric);
    }

    private VectorSearch withFullIvfProbe(VectorSearch vectorSearch) {
        Map<String, String> options = new LinkedHashMap<>(vectorSearch.options());
        options.remove(L_SEARCH_PARAMETER);
        options.put(NPROBE_PARAMETER, Integer.toString(ivfListCount()));
        VectorSearch fullProbeSearch =
                new VectorSearch(
                        vectorSearch.vector(), vectorSearch.limit(), vectorSearch.fieldName(), options);
        if (vectorSearch.includeRowIds() != null) {
            fullProbeSearch.withIncludeRowIds(vectorSearch.includeRowIds());
        }
        return fullProbeSearch;
    }

    private BatchVectorSearch withFullIvfProbe(BatchVectorSearch batchVectorSearch) {
        Map<String, String> options = new LinkedHashMap<>(batchVectorSearch.options());
        options.remove(L_SEARCH_PARAMETER);
        options.put(NPROBE_PARAMETER, Integer.toString(ivfListCount()));
        BatchVectorSearch fullProbeSearch =
                new BatchVectorSearch(
                        batchVectorSearch.vectors(),
                        batchVectorSearch.limit(),
                        batchVectorSearch.fieldName(),
                        options);
        if (batchVectorSearch.includeRowIds() != null) {
            fullProbeSearch.withIncludeRowIds(batchVectorSearch.includeRowIds());
        }
        return fullProbeSearch;
    }

    private int ivfListCount() {
        for (String methodName : new String[] {"nlist", "getNlist", "ivfListCount"}) {
            try {
                Method method = nativeMeta.getClass().getMethod(methodName);
                Object result = method.invoke(nativeMeta);
                if (result instanceof Number) {
                    int nlist = ((Number) result).intValue();
                    if (nlist > 0) {
                        return nlist;
                    }
                }
            } catch (NoSuchMethodException e) {
                // Try the next common metadata accessor name.
            } catch (IllegalAccessException e) {
                throw new RuntimeException("Failed to access native vector index nlist metadata.", e);
            } catch (InvocationTargetException e) {
                throw rethrowInvocation("Failed to read native vector index nlist metadata.", e);
            }
        }
        throw new IllegalStateException(
                "Centroid shard search mode 'all-lists' requires native metadata to expose nlist.");
    }

    private VectorSearchResult searchIvfPqCentroid(
            float[] queryVector, VectorSearchParams params, int centroid) {
        return invokeRequired(
                SEARCH_CENTROID_METHOD,
                new Class<?>[] {float[].class, VectorSearchParams.class, int.class},
                new Object[] {queryVector, params, centroid},
                VectorSearchResult.class);
    }

    private VectorSearchResult searchIvfPqCentroid(
            float[] queryVector, VectorSearchParams params, int centroid, byte[] filterBytes) {
        return invokeRequired(
                SEARCH_CENTROID_METHOD,
                new Class<?>[] {float[].class, VectorSearchParams.class, int.class, byte[].class},
                new Object[] {queryVector, params, centroid, filterBytes},
                VectorSearchResult.class);
    }

    private VectorSearchBatchResult searchIvfPqCentroidBatch(
            float[] queryVectors, int queryCount, VectorSearchParams params, int centroid) {
        return invokeRequired(
                SEARCH_CENTROID_BATCH_METHOD,
                new Class<?>[] {float[].class, int.class, VectorSearchParams.class, int.class},
                new Object[] {queryVectors, queryCount, params, centroid},
                VectorSearchBatchResult.class);
    }

    private VectorSearchBatchResult searchIvfPqCentroidBatch(
            float[] queryVectors,
            int queryCount,
            VectorSearchParams params,
            int centroid,
            byte[] filterBytes) {
        return invokeRequired(
                SEARCH_CENTROID_BATCH_METHOD,
                new Class<?>[] {
                    float[].class, int.class, VectorSearchParams.class, int.class, byte[].class
                },
                new Object[] {queryVectors, queryCount, params, centroid, filterBytes},
                VectorSearchBatchResult.class);
    }

    private <T> T invokeRequired(
            String methodName, Class<?>[] parameterTypes, Object[] args, Class<T> returnType) {
        try {
            Method method = vectorReader.getClass().getMethod(methodName, parameterTypes);
            Object result = method.invoke(vectorReader, args);
            return returnType.cast(result);
        } catch (NoSuchMethodException e) {
            throw new UnsupportedOperationException(
                    "Centroid shard search mode 'forced-centroid' requires native API "
                            + "VectorIndexReader."
                            + methodName
                            + "(...). Set option 'ivf.pq.centroid.search.mode=all-lists' "
                            + "to use the compatibility path.",
                    e);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(
                    "Failed to access native VectorIndexReader." + methodName + "(...).", e);
        } catch (InvocationTargetException e) {
            throw rethrowInvocation(
                    "Failed to invoke native VectorIndexReader." + methodName + "(...).", e);
        }
    }

    private static RuntimeException rethrowInvocation(String message, InvocationTargetException e) {
        Throwable cause = e.getCause();
        if (cause instanceof RuntimeException) {
            return (RuntimeException) cause;
        }
        if (cause instanceof Error) {
            throw (Error) cause;
        }
        return new RuntimeException(message, cause == null ? e : cause);
    }

    static Optional<ScoredGlobalIndexResult> buildScoredResult(
            long[] ids, float[] distances, String metric) {
        if (ids.length == 0) {
            return Optional.empty();
        }

        RoaringNavigableMap64 resultBitmap = new RoaringNavigableMap64();
        HashMap<Long, Float> id2scores = new HashMap<>(ids.length);

        for (int i = 0; i < ids.length; i++) {
            long rowId = ids[i];
            if (rowId < 0) {
                continue;
            }
            float score = convertDistanceToScore(distances[i], metric);
            resultBitmap.add(rowId);
            id2scores.put(rowId, score);
        }

        if (resultBitmap.isEmpty()) {
            return Optional.empty();
        }

        return Optional.of(
                ScoredGlobalIndexResult.create(
                        resultBitmap,
                        rowId -> {
                            Float score = id2scores.get(rowId);
                            if (score == null) {
                                throw new IllegalArgumentException(
                                        "No score found for rowId: "
                                                + rowId
                                                + ". Only rowIds present in results() are valid.");
                            }
                            return score;
                        }));
    }

    private static float[] flattenVectors(float[][] vectors, int vectorCount, int dimension) {
        float[] queries = new float[vectorCount * dimension];
        for (int i = 0; i < vectorCount; i++) {
            System.arraycopy(vectors[i], 0, queries, i * dimension, dimension);
        }
        return queries;
    }

    private static List<Optional<ScoredGlobalIndexResult>> emptyResults(int n) {
        List<Optional<ScoredGlobalIndexResult>> results = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            results.add(Optional.empty());
        }
        return results;
    }

    /** Resolves filter bytes and effective top-K; returns null when the filter selects no rows. */
    private static SearchScope resolveScope(RoaringNavigableMap64 includeRowIds, int limit)
            throws IOException {
        if (includeRowIds == null) {
            return new SearchScope(null, limit);
        }
        long cardinality = includeRowIds.getLongCardinality();
        if (cardinality == 0) {
            return null;
        }
        return new SearchScope(includeRowIds.serialize(), (int) Math.min(limit, cardinality));
    }

    /** Resolved filter state for a query. {@code filterBytes} is null when no filter is set. */
    private static final class SearchScope {
        private final byte[] filterBytes;
        private final int effectiveK;

        private SearchScope(byte[] filterBytes, int effectiveK) {
            this.filterBytes = filterBytes;
            this.effectiveK = effectiveK;
        }
    }

    private static float convertDistanceToScore(float distance, String metric) {
        if ("l2".equals(metric)) {
            return 1.0f / (1.0f + distance);
        } else if ("cosine".equals(metric)) {
            return 1.0f - distance;
        } else if ("inner_product".equals(metric)) {
            return -distance;
        }
        throw new IllegalArgumentException("Unknown metric: " + metric);
    }

    static VectorSearchParams searchParams(Map<String, String> parameters, int topK) {
        if (parameters == null) {
            parameters = Collections.emptyMap();
        }
        Integer nprobe = intParameter(parameters, NPROBE_PARAMETER);
        Integer lSearch = intParameter(parameters, L_SEARCH_PARAMETER);
        Integer maxInitialFilterExpansionFactor =
                intParameter(parameters, MAX_INITIAL_FILTER_EXPANSION_FACTOR_PARAMETER);
        if (nprobe != null && lSearch != null) {
            throw new IllegalArgumentException(
                    "Cannot set both '" + NPROBE_PARAMETER + "' and '" + L_SEARCH_PARAMETER + "'.");
        }
        VectorSearchParams searchParams;
        if (nprobe != null) {
            searchParams = VectorSearchParams.ivf(topK, nprobe);
        } else if (lSearch != null) {
            searchParams = VectorSearchParams.diskAnn(topK, lSearch);
        } else {
            searchParams = VectorSearchParams.automatic(topK);
        }
        return maxInitialFilterExpansionFactor == null
                ? searchParams
                : searchParams.withMaxInitialFilterExpansionFactor(maxInitialFilterExpansionFactor);
    }

    static VectorSearchParams batchSearchParams(Map<String, String> parameters, int topK) {
        VectorSearchParams searchParams = searchParams(parameters, topK);
        String reuseMode = parameters.get(IVF_PQ_BATCH_TABLE_REUSE_PARAMETER);
        if (reuseMode != null) {
            searchParams = searchParams.withIvfPqBatchTableReuse(reuseMode);
        }
        Long reuseMaxBytes =
                longParameter(parameters, IVF_PQ_BATCH_TABLE_REUSE_MAX_BYTES_PARAMETER);
        return reuseMaxBytes == null
                ? searchParams
                : searchParams.withIvfPqBatchTableReuseMaxBytes(reuseMaxBytes);
    }

    private static Integer intParameter(Map<String, String> parameters, String key) {
        String value = parameters.get(key);
        if (value == null) {
            return null;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid value for '" + key + "': " + value + ". Must be an integer.", e);
        }
    }

    private static Long longParameter(Map<String, String> parameters, String key) {
        String value = parameters.get(key);
        if (value == null) {
            return null;
        }
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid value for '" + key + "': " + value + ". Must be a long integer.", e);
        }
    }

    private void validateSearchVector(Object vector) {
        if (!(vector instanceof float[])) {
            throw new IllegalArgumentException(
                    "Expected float[] vector but got: " + vector.getClass());
        }
        boolean validFieldType = false;
        if (fieldType instanceof VectorType) {
            validFieldType = ((VectorType) fieldType).getElementType() instanceof FloatType;
        } else if (fieldType instanceof ArrayType) {
            validFieldType = ((ArrayType) fieldType).getElementType() instanceof FloatType;
        }
        if (!validFieldType) {
            throw new IllegalArgumentException(
                    "Vector index requires VectorType<FLOAT> or ArrayType<FLOAT>, but field type is: "
                            + fieldType);
        }
        int queryDim = ((float[]) vector).length;
        if (queryDim != nativeMeta.dimension()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Query vector dimension mismatch: index expects %d, but got %d",
                            nativeMeta.dimension(), queryDim));
        }
    }

    private void ensureLoaded() throws IOException {
        if (vectorReader == null) {
            synchronized (this) {
                if (vectorReader == null) {
                    SeekableInputStream in = fileReader.getInputStream(ioMeta);
                    VectorIndexReader reader = null;
                    try {
                        reader = new VectorIndexReader(new SeekableStreamVectorIndexInput(in));
                        nativeMeta = reader.metadata();
                        reader.optimizeForSearch();
                        vectorReader = reader;
                        openStream = in;
                    } catch (Exception e) {
                        if (reader != null) {
                            try {
                                reader.close();
                            } catch (Exception closeException) {
                                e.addSuppressed(closeException);
                            }
                        }
                        IOUtils.closeQuietly(in);
                        throw e;
                    }
                }
            }
        }
    }

    @Override
    public void close() throws IOException {
        Throwable firstException = null;

        if (vectorReader != null) {
            try {
                vectorReader.close();
            } catch (Throwable t) {
                firstException = t;
            }
            vectorReader = null;
        }

        if (openStream != null) {
            try {
                openStream.close();
            } catch (Throwable t) {
                if (firstException == null) {
                    firstException = t;
                } else {
                    firstException.addSuppressed(t);
                }
            }
            openStream = null;
        }

        if (firstException != null) {
            if (firstException instanceof IOException) {
                throw (IOException) firstException;
            } else if (firstException instanceof RuntimeException) {
                throw (RuntimeException) firstException;
            } else {
                throw new RuntimeException(
                        "Failed to close vector global index reader", firstException);
            }
        }
    }

    static class SeekableStreamVectorIndexInput implements VectorIndexInput {

        private final SeekableInputStream input;

        SeekableStreamVectorIndexInput(SeekableInputStream input) {
            this.input = input;
        }

        @Override
        public void pread(long[] positions, byte[][] buffers) {
            if (positions.length != buffers.length) {
                throw new IllegalArgumentException(
                        "positions length "
                                + positions.length
                                + " != buffers length "
                                + buffers.length);
            }
            try {
                if (input instanceof VectoredReadable
                        && areRangesNonOverlapping(positions, buffers)) {
                    preadVectored((VectoredReadable) input, positions, buffers);
                } else {
                    synchronized (this) {
                        preadSequential(positions, buffers);
                    }
                }
            } catch (IOException e) {
                throw new RuntimeException("Failed to read vector index", e);
            }
        }

        private void preadVectored(VectoredReadable readable, long[] positions, byte[][] buffers)
                throws IOException {
            List<FileRange> ranges = new ArrayList<>(positions.length);
            for (int i = 0; i < positions.length; i++) {
                ranges.add(FileRange.createFileRange(positions[i], buffers[i]));
            }

            VectoredReadUtils.ReadOptions options =
                    VectoredReadUtils.ReadOptions.from(readable)
                            .withMinSeekForVectorReads(VECTOR_INDEX_MIN_SEEK_FOR_VECTOR_READS)
                            .withParallelismForVectorReads(
                                    VECTOR_INDEX_PARALLELISM_FOR_VECTOR_READS)
                            .withSequentialReadFallback(false);
            VectoredReadUtils.readVectored(readable, ranges, options);

            for (FileRange range : ranges) {
                range.getData().join();
            }
        }

        private void preadSequential(long[] positions, byte[][] buffers) throws IOException {
            for (int i = 0; i < positions.length; i++) {
                input.seek(positions[i]);
                readFully(input, buffers[i]);
            }
        }

        private static void readFully(SeekableInputStream input, byte[] buffer) throws IOException {
            int offset = 0;
            while (offset < buffer.length) {
                int read = input.read(buffer, offset, buffer.length - offset);
                if (read < 0) {
                    throw new IOException("Unexpected end of vector index file");
                }
                offset += read;
            }
        }

        private static boolean areRangesNonOverlapping(long[] positions, byte[][] buffers) {
            if (positions.length < 2) {
                return true;
            }

            List<Integer> indexes = new ArrayList<>(positions.length);
            for (int i = 0; i < positions.length; i++) {
                indexes.add(i);
            }
            indexes.sort(Comparator.comparingLong(index -> positions[index]));

            boolean hasPrevious = false;
            long previousEnd = 0;
            for (int index : indexes) {
                long offset = positions[index];
                long end = offset + buffers[index].length;
                if (end < offset || (hasPrevious && offset < previousEnd)) {
                    return false;
                }
                previousEnd = end;
                hasPrevious = true;
            }
            return true;
        }
    }

    // =================== unsupported =====================

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitIsNotNull(FieldRef fieldRef) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitIsNull(FieldRef fieldRef) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitStartsWith(
            FieldRef fieldRef, Object literal) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitEndsWith(
            FieldRef fieldRef, Object literal) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitContains(
            FieldRef fieldRef, Object literal) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitLike(
            FieldRef fieldRef, Object literal) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitLessThan(
            FieldRef fieldRef, Object literal) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitGreaterOrEqual(
            FieldRef fieldRef, Object literal) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitNotEqual(
            FieldRef fieldRef, Object literal) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitLessOrEqual(
            FieldRef fieldRef, Object literal) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitEqual(
            FieldRef fieldRef, Object literal) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitGreaterThan(
            FieldRef fieldRef, Object literal) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitIn(
            FieldRef fieldRef, List<Object> literals) {
        return CompletableFuture.completedFuture(Optional.empty());
    }

    @Override
    public CompletableFuture<Optional<GlobalIndexResult>> visitNotIn(
            FieldRef fieldRef, List<Object> literals) {
        return CompletableFuture.completedFuture(Optional.empty());
    }
}
