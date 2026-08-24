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

import org.apache.paimon.globalindex.GlobalIndexIOMeta;
import org.apache.paimon.globalindex.GlobalIndexReader;
import org.apache.paimon.globalindex.GlobalIndexResult;
import org.apache.paimon.globalindex.ScoredGlobalIndexResult;
import org.apache.paimon.globalindex.io.GlobalIndexFileReader;
import org.apache.paimon.predicate.BatchVectorSearch;
import org.apache.paimon.predicate.FieldRef;
import org.apache.paimon.predicate.VectorSearch;
import org.apache.paimon.types.DataType;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;

/** Reader that routes a vector search to centroid data shard files before ANN search. */
public class NativeCentroidRoutedIndexReader implements GlobalIndexReader {

    static final String CENTROID_SHARD_SEARCH_MODE_OPTION = "ivf.pq.centroid.search.mode";
    private static final String FORCED_CENTROID_SEARCH_MODE = "forced-centroid";
    private static final String ALL_LISTS_SEARCH_MODE = "all-lists";

    private final GlobalIndexFileReader fileReader;
    @Nullable private final NativeVectorCentroidRouter centroidRouter;
    private final Map<Integer, GlobalIndexIOMeta> centroidToFile;
    private final DataType fieldType;
    private final ExecutorService executor;
    private final Map<Integer, NativeVectorGlobalIndexReader> openedReaders;
    private final CentroidShardSearchMode defaultSearchMode;

    public NativeCentroidRoutedIndexReader(
            GlobalIndexFileReader fileReader,
            @Nullable NativeVectorCentroidRouter centroidRouter,
            Map<Integer, GlobalIndexIOMeta> centroidToFile,
            DataType fieldType,
            ExecutorService executor,
            Map<String, String> options) {
        this.fileReader = fileReader;
        this.centroidRouter = centroidRouter;
        this.centroidToFile = centroidToFile;
        this.fieldType = fieldType;
        this.executor = executor;
        this.openedReaders = new HashMap<>();
        this.defaultSearchMode =
                CentroidShardSearchMode.fromOptions(options, CentroidShardSearchMode.FORCED_CENTROID);
    }

    public Optional<ScoredGlobalIndexResult> search(VectorSearch vectorSearch) throws IOException {
        Iterable<Integer> centroids = routedCentroids(vectorSearch);
        List<ScoredGlobalIndexResult> partialResults = new ArrayList<>();
        CentroidShardSearchMode searchMode =
                CentroidShardSearchMode.fromOptions(vectorSearch.options(), defaultSearchMode);
        for (int centroid : centroids) {
            GlobalIndexIOMeta file = centroidToFile.get(centroid);
            if (file == null) {
                continue;
            }
            NativeVectorGlobalIndexReader reader = getReader(centroid, file);
            Optional<ScoredGlobalIndexResult> result =
                    reader.searchCentroidShard(vectorSearch, centroid, searchMode);
            if (result.isPresent()) {
                partialResults.add(result.get());
            }
        }
        if (partialResults.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(ScoredGlobalIndexResult.merge(partialResults).topK(vectorSearch.limit()));
    }

    private List<Optional<ScoredGlobalIndexResult>> searchBatch(BatchVectorSearch batchVectorSearch)
            throws IOException {
        int n = batchVectorSearch.vectorCount();
        if (n == 1) {
            List<Optional<ScoredGlobalIndexResult>> results = new ArrayList<>(1);
            results.add(search(batchVectorSearch.forIndex(0)));
            return results;
        }

        CentroidShardSearchMode searchMode =
                CentroidShardSearchMode.fromOptions(batchVectorSearch.options(), defaultSearchMode);
        Map<Integer, List<Integer>> centroidToQueryIndexes = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) {
            for (int centroid : routedCentroids(batchVectorSearch.forIndex(i))) {
                if (!centroidToFile.containsKey(centroid)) {
                    continue;
                }
                List<Integer> queryIndexes = centroidToQueryIndexes.get(centroid);
                if (queryIndexes == null) {
                    queryIndexes = new ArrayList<>();
                    centroidToQueryIndexes.put(centroid, queryIndexes);
                }
                queryIndexes.add(i);
            }
        }

        List<List<ScoredGlobalIndexResult>> partialResultsByQuery = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            partialResultsByQuery.add(new ArrayList<>());
        }

        for (Map.Entry<Integer, List<Integer>> entry : centroidToQueryIndexes.entrySet()) {
            int centroid = entry.getKey();
            List<Integer> queryIndexes = entry.getValue();
            NativeVectorGlobalIndexReader reader = getReader(centroid, centroidToFile.get(centroid));
            BatchVectorSearch shardBatchSearch = subBatchSearch(batchVectorSearch, queryIndexes);
            List<Optional<ScoredGlobalIndexResult>> shardResults =
                    reader.searchCentroidShardBatch(shardBatchSearch, centroid, searchMode);
            for (int i = 0; i < queryIndexes.size(); i++) {
                Optional<ScoredGlobalIndexResult> result = shardResults.get(i);
                if (result.isPresent()) {
                    partialResultsByQuery.get(queryIndexes.get(i)).add(result.get());
                }
            }
        }

        List<Optional<ScoredGlobalIndexResult>> results = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            List<ScoredGlobalIndexResult> partialResults = partialResultsByQuery.get(i);
            if (partialResults.isEmpty()) {
                results.add(Optional.empty());
            } else {
                results.add(
                        Optional.of(
                                ScoredGlobalIndexResult.merge(partialResults)
                                        .topK(batchVectorSearch.limit())));
            }
        }
        return results;
    }

    private BatchVectorSearch subBatchSearch(
            BatchVectorSearch batchVectorSearch, List<Integer> queryIndexes) {
        float[][] vectors = new float[queryIndexes.size()][];
        float[][] allVectors = batchVectorSearch.vectors();
        for (int i = 0; i < queryIndexes.size(); i++) {
            vectors[i] = allVectors[queryIndexes.get(i)];
        }

        BatchVectorSearch subBatchSearch =
                new BatchVectorSearch(
                        vectors,
                        batchVectorSearch.limit(),
                        batchVectorSearch.fieldName(),
                        batchVectorSearch.options());
        if (batchVectorSearch.includeRowIds() != null) {
            subBatchSearch.withIncludeRowIds(batchVectorSearch.includeRowIds());
        }
        return subBatchSearch;
    }

    private Iterable<Integer> routedCentroids(VectorSearch vectorSearch) throws IOException {
        if (centroidRouter == null) {
            return centroidToFile.keySet();
        }
        List<Integer> result = new ArrayList<>();
        for (int centroid : centroidRouter.route(vectorSearch)) {
            result.add(centroid);
        }
        return result;
    }

    private synchronized NativeVectorGlobalIndexReader getReader(
            int centroid, GlobalIndexIOMeta file) {
        NativeVectorGlobalIndexReader reader = openedReaders.get(centroid);
        if (reader == null) {
            reader =
                    new NativeVectorGlobalIndexReader(
                            fileReader, Collections.singletonList(file), fieldType, executor);
            openedReaders.put(centroid, reader);
        }
        return reader;
    }

    @Override
    public boolean returnsAbsoluteRowIds() {
        return true;
    }

    @Override
    public CompletableFuture<Optional<ScoredGlobalIndexResult>> visitVectorSearch(
            VectorSearch vectorSearch) {
        try {
            return CompletableFuture.completedFuture(search(vectorSearch));
        } catch (IOException e) {
            CompletableFuture<Optional<ScoredGlobalIndexResult>> failed = new CompletableFuture<>();
            failed.completeExceptionally(
                    new RuntimeException("Failed centroid-routed vector index search", e));
            return failed;
        }
    }

    @Override
    public CompletableFuture<List<Optional<ScoredGlobalIndexResult>>> visitBatchVectorSearch(
            BatchVectorSearch batchVectorSearch) {
        try {
            return CompletableFuture.completedFuture(searchBatch(batchVectorSearch));
        } catch (IOException e) {
            CompletableFuture<List<Optional<ScoredGlobalIndexResult>>> failed =
                    new CompletableFuture<>();
            failed.completeExceptionally(
                    new RuntimeException("Failed centroid-routed batch vector index search", e));
            return failed;
        }
    }

    @Override
    public void close() throws IOException {
        Throwable firstException = null;
        if (centroidRouter != null) {
            try {
                centroidRouter.close();
            } catch (Throwable t) {
                firstException = t;
            }
        }
        for (NativeVectorGlobalIndexReader reader : openedReaders.values()) {
            try {
                reader.close();
            } catch (Throwable t) {
                if (firstException == null) {
                    firstException = t;
                } else {
                    firstException.addSuppressed(t);
                }
            }
        }
        openedReaders.clear();

        if (firstException != null) {
            if (firstException instanceof IOException) {
                throw (IOException) firstException;
            }
            if (firstException instanceof RuntimeException) {
                throw (RuntimeException) firstException;
            }
            throw new RuntimeException(
                    "Failed to close centroid-routed vector index reader", firstException);
        }
    }

    enum CentroidShardSearchMode {
        FORCED_CENTROID,
        ALL_LISTS;

        static CentroidShardSearchMode fromOptions(
                Map<String, String> options, CentroidShardSearchMode defaultMode) {
            if (options == null) {
                return defaultMode;
            }
            String value = options.get(CENTROID_SHARD_SEARCH_MODE_OPTION);
            if (value == null || value.trim().isEmpty()) {
                return defaultMode;
            }
            String normalized = value.trim().toLowerCase().replace('_', '-');
            if (FORCED_CENTROID_SEARCH_MODE.equals(normalized)) {
                return FORCED_CENTROID;
            }
            if (ALL_LISTS_SEARCH_MODE.equals(normalized)) {
                return ALL_LISTS;
            }
            throw new IllegalArgumentException(
                    "Invalid value for '"
                            + CENTROID_SHARD_SEARCH_MODE_OPTION
                            + "': "
                            + value
                            + ". Supported values are '"
                            + FORCED_CENTROID_SEARCH_MODE
                            + "' and '"
                            + ALL_LISTS_SEARCH_MODE
                            + "'.");
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
