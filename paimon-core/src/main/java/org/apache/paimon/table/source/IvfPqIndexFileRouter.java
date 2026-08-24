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

package org.apache.paimon.table.source;

import org.apache.paimon.Snapshot;
import org.apache.paimon.data.BinaryRow;
import org.apache.paimon.globalindex.GlobalIndexIOMeta;
import org.apache.paimon.globalindex.VectorGlobalIndexer;
import org.apache.paimon.globalindex.io.GlobalIndexFileReader;
import org.apache.paimon.index.GlobalIndexMeta;
import org.apache.paimon.index.IndexFileHandler;
import org.apache.paimon.index.IndexFileMeta;
import org.apache.paimon.index.IndexPathFactory;
import org.apache.paimon.manifest.IndexManifestEntry;
import org.apache.paimon.partition.PartitionPredicate;
import org.apache.paimon.predicate.Predicate;
import org.apache.paimon.table.FileStoreTable;
import org.apache.paimon.types.DataField;

import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.apache.paimon.utils.Preconditions.checkNotNull;

/** Planner-side index file router for centroid-based IVF_PQ vector indexes. */
class IvfPqIndexFileRouter {

    private static final String IVF_PQ_INDEX_TYPE = "ivf-pq";

    private final FileStoreTable table;
    @Nullable private final PartitionPredicate partitionFilter;
    @Nullable private final Predicate filter;
    private final DataField vectorColumn;
    private final Map<String, String> options;
    @Nullable private final float[][] queryVectors;
    private final int limit;
    @Nullable private final String indexType;
    @Nullable private final VectorGlobalIndexer vectorGlobalIndexer;

    IvfPqIndexFileRouter(
            FileStoreTable table,
            @Nullable PartitionPredicate partitionFilter,
            @Nullable Predicate filter,
            DataField vectorColumn,
            Map<String, String> options,
            @Nullable float[][] queryVectors,
            int limit,
            @Nullable String indexType,
            @Nullable VectorGlobalIndexer vectorGlobalIndexer) {
        this.table = table;
        this.partitionFilter = partitionFilter;
        this.filter = filter;
        this.vectorColumn = vectorColumn;
        this.options = options;
        this.queryVectors = queryVectors;
        this.limit = limit;
        this.indexType = indexType;
        this.vectorGlobalIndexer = vectorGlobalIndexer;
    }

    List<IndexManifestEntry> searchableIndexFileEntries(List<IndexManifestEntry> indexFiles) {
        return indexFiles.stream()
                .filter(f -> !isRoutingGlobalIndexFile(f.indexFile()))
                .collect(Collectors.toList());
    }

    @Nullable
    List<IndexFileMeta> tryRouteIndexFiles(
            @Nullable Snapshot snapshot,
            IndexFileHandler indexFileHandler,
            List<IndexManifestEntry> vectorAndScalarCandidateEntries) {
        if (!canRoute()) {
            return null;
        }

        IndexPathFactory pathFactory = table.store().pathFactory().globalIndexFileFactory();
        GlobalIndexFileReader fileReader = m -> table.fileIO().newInputStream(m.filePath());

        Set<BinaryRow> candidateVectorPartitions =
                candidateVectorPartitions(vectorAndScalarCandidateEntries);
        if (candidateVectorPartitions.isEmpty()) {
            return null;
        }

        Map<BinaryRow, Set<Integer>> routedCentroidsByPartition = new HashMap<>();
        Map<BinaryRow, List<GlobalIndexIOMeta>> globalIndexFilesByPartition =
                routingGlobalIndexFilesByPartition(snapshot, indexFileHandler, pathFactory);
        if (globalIndexFilesByPartition.isEmpty()) {
            return null;
        }
        for (BinaryRow partition : candidateVectorPartitions) {
            List<GlobalIndexIOMeta> globalIndexFiles = globalIndexFilesByPartition.get(partition);
            if (globalIndexFiles == null || globalIndexFiles.isEmpty()) {
                return null;
            }
            for (GlobalIndexIOMeta globalIndexFile : globalIndexFiles) {
                routedCentroidsByPartition
                        .computeIfAbsent(partition, k -> new HashSet<>())
                        .addAll(
                                checkNotNull(vectorGlobalIndexer)
                                        .routeCentroids(
                                                fileReader,
                                                globalIndexFile,
                                                checkNotNull(queryVectors),
                                                limit,
                                                options));
            }
        }

        List<IndexFileMeta> routedIndexFiles = new ArrayList<>();
        for (IndexManifestEntry entry : vectorAndScalarCandidateEntries) {
            GlobalIndexMeta globalIndex = checkNotNull(entry.indexFile().globalIndexMeta());
            if (!isPrimaryColumn(globalIndex, vectorColumn.id())) {
                continue;
            }
            Set<Integer> routedCentroids = routedCentroidsByPartition.get(entry.partition());
            if (routedCentroids == null) {
                continue;
            }
            if (checkNotNull(vectorGlobalIndexer)
                    .acceptsRoutedIndexFile(globalIndex.indexMeta(), routedCentroids)) {
                routedIndexFiles.add(entry.indexFile());
            }
        }
        return routedIndexFiles;
    }

    private boolean canRoute() {
        if (queryVectors == null || queryVectors.length == 0 || limit <= 0) {
            return false;
        }
        // The first planner-side centroid routing implementation only handles the pure
        // partition-pruning case. When there are additional scalar predicates, keep the original
        // planning flow so scalar pre-filtering semantics are not mixed with centroid routing.
        if (filter != null) {
            return false;
        }
        if (!IVF_PQ_INDEX_TYPE.equals(indexType)) {
            return false;
        }
        return vectorGlobalIndexer != null;
    }

    private Set<BinaryRow> candidateVectorPartitions(
            List<IndexManifestEntry> vectorAndScalarCandidateEntries) {
        Set<BinaryRow> result = new HashSet<>();
        for (IndexManifestEntry entry : vectorAndScalarCandidateEntries) {
            IndexFileMeta indexFile = entry.indexFile();
            GlobalIndexMeta globalIndex = checkNotNull(indexFile.globalIndexMeta());
            if (!isPrimaryColumn(globalIndex, vectorColumn.id())
                    || isRoutingGlobalIndexFile(indexFile)) {
                continue;
            }
            result.add(entry.partition());
        }
        return result;
    }

    private Map<BinaryRow, List<GlobalIndexIOMeta>> routingGlobalIndexFilesByPartition(
            @Nullable Snapshot snapshot,
            IndexFileHandler indexFileHandler,
            IndexPathFactory pathFactory) {
        Map<BinaryRow, List<GlobalIndexIOMeta>> result = new HashMap<>();
        for (IndexManifestEntry entry :
                indexFileHandler.scan(snapshot, this::isRoutingGlobalIndexFileEntry)) {
            IndexFileMeta indexFile = entry.indexFile();
            GlobalIndexMeta globalIndex = checkNotNull(indexFile.globalIndexMeta());
            result.computeIfAbsent(entry.partition(), k -> new ArrayList<>())
                    .add(
                            new GlobalIndexIOMeta(
                                    pathFactory.toPath(indexFile),
                                    indexFile.fileSize(),
                                    globalIndex.indexMeta(),
                                    indexFile.fileKind()));
        }
        return result;
    }

    private boolean isRoutingGlobalIndexFileEntry(IndexManifestEntry entry) {
        if (partitionFilter != null && !partitionFilter.test(entry.partition())) {
            return false;
        }
        if (!checkNotNull(indexType).equals(entry.indexType())) {
            return false;
        }
        IndexFileMeta indexFile = entry.indexFile();
        GlobalIndexMeta globalIndex = indexFile.globalIndexMeta();
        return globalIndex != null
                && isPrimaryColumn(globalIndex, vectorColumn.id())
                && isRoutingGlobalIndexFile(indexFile);
    }

    private boolean isRoutingGlobalIndexFile(IndexFileMeta indexFile) {
        if (vectorGlobalIndexer == null) {
            return false;
        }
        GlobalIndexMeta globalIndex = indexFile.globalIndexMeta();
        return globalIndex != null
                && vectorGlobalIndexer.isRoutingGlobalIndexFile(
                        indexFile.fileKind(), globalIndex.indexMeta());
    }

    private static boolean isPrimaryColumn(GlobalIndexMeta meta, int fieldId) {
        return meta.indexFieldId() == fieldId;
    }
}
