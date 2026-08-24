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

import org.apache.paimon.Snapshot;
import org.apache.paimon.data.BinaryRow;
import org.apache.paimon.globalindex.GlobalIndexBuilderUtils;
import org.apache.paimon.globalindex.IndexedSplit;
import org.apache.paimon.globalindex.IvfPqShard;
import org.apache.paimon.index.GlobalIndexMeta;
import org.apache.paimon.io.DataFileMeta;
import org.apache.paimon.manifest.IndexManifestEntry;
import org.apache.paimon.manifest.ManifestEntry;
import org.apache.paimon.partition.PartitionPredicate;
import org.apache.paimon.schema.SchemaManager;
import org.apache.paimon.table.FileStoreTable;
import org.apache.paimon.table.source.DataSplit;
import org.apache.paimon.types.DataField;
import org.apache.paimon.utils.Range;

import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.apache.paimon.utils.Preconditions.checkArgument;
import static org.apache.paimon.vector.index.NativeVectorIndexOptions.IVF_PQ_SHARD_OPTION;

/** Plans and validates centroid-sharded IVF/PQ global index build inputs. */
class CentroidShardedIvfPqIndexBuildPlanner {

    private final List<DataField> indexFields;
    private final List<Range> rowRangesToBuild;
    private final List<IndexedSplit> splits;

    private CentroidShardedIvfPqIndexBuildPlanner(
            List<DataField> indexFields, List<Range> rowRangesToBuild, List<IndexedSplit> splits) {
        this.indexFields = indexFields;
        this.rowRangesToBuild = rowRangesToBuild;
        this.splits = splits;
    }

    static CentroidShardedIvfPqIndexBuildPlanner create(
            FileStoreTable table,
            Snapshot snapshot,
            String indexType,
            DataField indexField,
            @Nullable PartitionPredicate partitionPredicate) {
        List<ManifestEntry> entries =
                table.store()
                        .newScan()
                        .withSnapshot(snapshot)
                        .withPartitionFilter(partitionPredicate)
                        .plan()
                        .files();
        List<DataField> indexFields = Collections.singletonList(indexField);
        List<String> indexColumns = Collections.singletonList(indexField.name());
        SchemaManager schemaManager = new SchemaManager(table.fileIO(), table.location());

        validateNoExistingIndexFiles(table, snapshot, indexType, indexField, partitionPredicate);
        long boundaryRowId =
                GlobalIndexBuilderUtils.findMinNonIndexableRowId(
                        schemaManager, entries, indexColumns);
        checkArgument(
                boundaryRowId == Long.MAX_VALUE,
                "The centroid-sharded implementation of '%s=%s' currently supports only "
                        + "full index builds for one partition. Some files cannot be indexed "
                        + "from row id %s because they do not contain all index columns.",
                IVF_PQ_SHARD_OPTION,
                IvfPqShard.CENTROID_BASED.optionValue(),
                boundaryRowId);

        List<Range> rowRangesToBuild =
                GlobalIndexBuilderUtils.unindexedRowRanges(
                        table, snapshot, indexType, indexFields, partitionPredicate);
        if (rowRangesToBuild.isEmpty()) {
            return new CentroidShardedIvfPqIndexBuildPlanner(
                    indexFields, rowRangesToBuild, Collections.emptyList());
        }
        validateFullRebuild(snapshot, rowRangesToBuild);

        List<IndexedSplit> splits = createPartitionIndexedSplits(table, entries, rowRangesToBuild);
        validateSinglePartition(splits);
        return new CentroidShardedIvfPqIndexBuildPlanner(indexFields, rowRangesToBuild, splits);
    }

    List<DataField> indexFields() {
        return indexFields;
    }

    List<Range> rowRangesToBuild() {
        return rowRangesToBuild;
    }

    List<IndexedSplit> splits() {
        return splits;
    }

    boolean isEmpty() {
        return splits.isEmpty();
    }

    private static void validateNoExistingIndexFiles(
            FileStoreTable table,
            Snapshot snapshot,
            String indexType,
            DataField indexField,
            @Nullable PartitionPredicate partitionPredicate) {
        List<String> existingIndexFiles = new ArrayList<>();
        for (IndexManifestEntry entry :
                table.store().newIndexFileHandler().scan(snapshot, indexType)) {
            if (partitionPredicate != null && !partitionPredicate.test(entry.partition())) {
                continue;
            }
            GlobalIndexMeta meta = entry.indexFile().globalIndexMeta();
            if (meta == null
                    || meta.indexFieldId() != indexField.id()
                    || hasExtraFieldIds(meta.extraFieldIds())) {
                continue;
            }
            existingIndexFiles.add(entry.indexFile().fileName());
        }

        checkArgument(
                existingIndexFiles.isEmpty(),
                "The centroid-sharded implementation of '%s=%s' supports only full rebuilds "
                        + "without existing valid index files. Found existing index files for "
                        + "index type '%s' on field '%s': %s. Please drop the existing index "
                        + "first, then rebuild it.",
                IVF_PQ_SHARD_OPTION,
                IvfPqShard.CENTROID_BASED.optionValue(),
                indexType,
                indexField.name(),
                existingIndexFiles);
    }

    private static boolean hasExtraFieldIds(@Nullable int[] extraFieldIds) {
        return extraFieldIds != null && extraFieldIds.length > 0;
    }

    private static void validateFullRebuild(Snapshot snapshot, List<Range> rowRangesToBuild) {
        long fullRangeEnd = snapshot.nextRowId() - 1;
        checkArgument(
                rowRangesToBuild.size() == 1
                        && rowRangesToBuild.get(0).from == 0
                        && rowRangesToBuild.get(0).to == fullRangeEnd,
                "The centroid-sharded implementation of '%s=%s' currently supports only "
                        + "full index builds for one partition. Incremental or partial builds "
                        + "are not supported. Row ranges to build: %s, expected: [%s, %s].",
                IVF_PQ_SHARD_OPTION,
                IvfPqShard.CENTROID_BASED.optionValue(),
                rowRangesToBuild,
                0,
                fullRangeEnd);
    }

    private static void validateSinglePartition(List<IndexedSplit> splits) {
        BinaryRow partition = null;
        for (IndexedSplit split : splits) {
            if (partition == null) {
                partition = split.dataSplit().partition();
            } else {
                checkArgument(
                        partition.equals(split.dataSplit().partition()),
                        "The centroid-sharded implementation of '%s=%s' currently supports "
                                + "building only one partition at a time.",
                        IVF_PQ_SHARD_OPTION,
                        IvfPqShard.CENTROID_BASED.optionValue());
            }
        }
    }

    private static List<IndexedSplit> createPartitionIndexedSplits(
            FileStoreTable table, List<ManifestEntry> entries, List<Range> rowRangesToBuild) {
        Map<BinaryRow, Map<Integer, List<ManifestEntry>>> entriesByPartitionAndBucket =
                new LinkedHashMap<>();
        for (ManifestEntry entry : entries) {
            if (entry.file().firstRowId() == null) {
                continue;
            }
            entriesByPartitionAndBucket
                    .computeIfAbsent(entry.partition(), key -> new LinkedHashMap<>())
                    .computeIfAbsent(entry.bucket(), key -> new ArrayList<>())
                    .add(entry);
        }

        List<IndexedSplit> result = new ArrayList<>();
        for (Map.Entry<BinaryRow, Map<Integer, List<ManifestEntry>>> partitionEntry :
                entriesByPartitionAndBucket.entrySet()) {
            BinaryRow partition = partitionEntry.getKey();
            for (Map.Entry<Integer, List<ManifestEntry>> bucketEntry :
                    partitionEntry.getValue().entrySet()) {
                List<ManifestEntry> bucketEntries = bucketEntry.getValue();
                if (bucketEntries.isEmpty()) {
                    continue;
                }
                List<DataFileMeta> dataFiles =
                        bucketEntries.stream().map(ManifestEntry::file).collect(Collectors.toList());
                DataSplit dataSplit =
                        DataSplit.builder()
                                .withPartition(partition)
                                .withBucket(bucketEntry.getKey())
                                .withTotalBuckets(bucketEntries.get(0).totalBuckets())
                                .withDataFiles(dataFiles)
                                .withBucketPath(
                                        table.store()
                                                .pathFactory()
                                                .bucketPath(partition, bucketEntry.getKey())
                                                .toString())
                                .rawConvertible(false)
                                .build();
                result.add(new IndexedSplit(dataSplit, rowRangesToBuild, null));
            }
        }
        return result;
    }
}
