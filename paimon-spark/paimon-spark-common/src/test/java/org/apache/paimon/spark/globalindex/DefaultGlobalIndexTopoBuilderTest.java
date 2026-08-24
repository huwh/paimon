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

import org.apache.paimon.data.BinaryRow;
import org.apache.paimon.globalindex.IndexedSplit;
import org.apache.paimon.io.DataFileMeta;
import org.apache.paimon.options.Options;
import org.apache.paimon.spark.globalindex.sorted.SortedIndexTopoBuilder;
import org.apache.paimon.stats.SimpleStats;
import org.apache.paimon.table.source.DataSplit;
import org.apache.paimon.types.DataField;
import org.apache.paimon.types.IntType;
import org.apache.paimon.utils.Range;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.apache.paimon.CoreOptions.GLOBAL_INDEX_BUILD_MAX_PARALLELISM;
import static org.apache.paimon.CoreOptions.GLOBAL_INDEX_ROW_COUNT_PER_SHARD;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Tests for {@link DefaultGlobalIndexTopoBuilder}. */
public class DefaultGlobalIndexTopoBuilderTest {

    @Test
    void testBitmapUsesSortedTopologyBuilder() {
        assertThat(GlobalIndexTopologyBuilderUtils.createTopoBuilder("bitmap"))
                .isInstanceOf(SortedIndexTopoBuilder.class);
    }

    @Test
    void testRowsPerShardUsesMergedBuildOptions() {
        Map<String, String> tableOptions = new HashMap<>();
        tableOptions.put(GLOBAL_INDEX_ROW_COUNT_PER_SHARD.key(), "1000");
        Map<String, String> buildOptions = new HashMap<>();
        buildOptions.put(GLOBAL_INDEX_ROW_COUNT_PER_SHARD.key(), "25");

        assertThat(
                        DefaultGlobalIndexTopoBuilder.rowsPerShard(
                                new Options(tableOptions, buildOptions)))
                .isEqualTo(25L);
    }

    @Test
    void testParallelismUsesBuildMaxParallelism() {
        Options options =
                new Options(
                        Collections.singletonMap(GLOBAL_INDEX_BUILD_MAX_PARALLELISM.key(), "2"));

        assertThat(DefaultGlobalIndexTopoBuilder.parallelism(5, options)).isEqualTo(2);
        assertThat(DefaultGlobalIndexTopoBuilder.parallelism(1, options)).isEqualTo(1);
    }

    @Test
    void testRowsPerShardMustBePositive() {
        Options options =
                new Options(Collections.singletonMap(GLOBAL_INDEX_ROW_COUNT_PER_SHARD.key(), "0"));

        assertThatThrownBy(() -> DefaultGlobalIndexTopoBuilder.rowsPerShard(options))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(
                        "Option 'global-index.row-count-per-shard' must be greater than 0.");
    }

    @Test
    void testMaxParallelismMustBePositive() {
        Options options =
                new Options(
                        Collections.singletonMap(GLOBAL_INDEX_BUILD_MAX_PARALLELISM.key(), "0"));

        assertThatThrownBy(() -> DefaultGlobalIndexTopoBuilder.parallelism(5, options))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining(
                        "Option 'global-index.build.max-parallelism' must be greater than 0.");
    }

    @Test
    void testLimitSplitsToTrainingSampleRowsSelectsFilesByFileMetaRowCount() {
        IndexedSplit split =
                indexedSplit(
                        "bucket-0",
                        dataFile("file-0", 12, 0),
                        dataFile("file-1", 12, 12),
                        dataFile("file-2", 12, 24),
                        dataFile("file-3", 12, 36));

        List<IndexedSplit> sampledSplits =
                DefaultGlobalIndexTopoBuilder.limitSplitsToTrainingSampleRows(
                        Collections.singletonList(split), 30);

        assertThat(sampledSplits).hasSize(1);
        assertThat(fileNames(sampledSplits.get(0)))
                .containsExactly("file-0", "file-1", "file-2");
    }

    @Test
    void testTrainingSampleRowsUsesFieldOptionFirst() {
        Map<String, String> optionMap = new HashMap<>();
        optionMap.put("ivf-pq.train.sample-rows", "100");
        optionMap.put("fields.vec.train.sample-rows", "30");
        Options options = new Options(optionMap);

        assertThat(
                        DefaultGlobalIndexTopoBuilder.trainingSampleRows(
                                "ivf-pq", new DataField(0, "vec", new IntType()), options))
                .isEqualTo(30L);
    }

    private static IndexedSplit indexedSplit(String bucketPath, DataFileMeta... dataFiles) {
        DataSplit dataSplit =
                DataSplit.builder()
                        .withPartition(BinaryRow.EMPTY_ROW)
                        .withBucket(0)
                        .withTotalBuckets(1)
                        .withDataFiles(Arrays.asList(dataFiles))
                        .withBucketPath(bucketPath)
                        .rawConvertible(false)
                        .build();
        return new IndexedSplit(dataSplit, Collections.singletonList(new Range(0, 100)), null);
    }

    private static DataFileMeta dataFile(String fileName, long rowCount, long firstRowId) {
        return DataFileMeta.forAppend(
                fileName,
                0,
                rowCount,
                SimpleStats.EMPTY_STATS,
                0,
                0,
                0,
                Collections.emptyList(),
                null,
                null,
                null,
                null,
                firstRowId,
                null);
    }

    private static List<String> fileNames(IndexedSplit split) {
        return split.dataSplit().dataFiles().stream()
                .map(DataFileMeta::fileName)
                .collect(Collectors.toList());
    }
}
