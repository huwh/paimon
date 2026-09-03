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

package org.apache.paimon.globalindex;

import org.apache.paimon.globalindex.io.GlobalIndexFileReader;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

/** A {@link GlobalIndexer} that supports vector similarity search. */
public interface VectorGlobalIndexer extends GlobalIndexer {

    /** Returns the metric name used to convert vector distances to comparable scores. */
    String metric();

    /**
     * Returns whether a manifest-level index file and metadata payload represent a routing file.
     */
    default boolean isRoutingGlobalIndexFile(IndexFileKind fileKind, byte[] indexMeta) {
        return fileKind == IndexFileKind.ROUTING_MODEL;
    }

    /**
     * Routes query vectors through a persisted global routing file.
     *
     * <p>The default implementation means that the indexer does not support query-time split
     * pruning. Implementations may return centroids that should be used to select physical index
     * files from manifest metadata.
     */
    default Set<Integer> routeCentroids(
            GlobalIndexFileReader fileReader,
            GlobalIndexIOMeta globalIndexFile,
            float[][] queryVectors,
            int limit,
            Map<String, String> options) {
        return Collections.emptySet();
    }

    /** Returns whether an index file metadata payload belongs to one of the routed centroids. */
    default boolean acceptsRoutedIndexFile(byte[] indexMeta, Set<Integer> routedCentroids) {
        return false;
    }

    /**
     * Returns whether an index file belongs to a routed shard of the supplied routing model.
     *
     * <p>The routing metadata parameter lets implementations reject data files produced by a
     * different model generation, even when both generations use the same shard ids.
     */
    default boolean acceptsRoutedIndexFile(
            byte[] routingIndexMeta, byte[] indexMeta, Set<Integer> routedCentroids) {
        return acceptsRoutedIndexFile(indexMeta, routedCentroids);
    }

    /** Returns whether an index data file was produced from the supplied routing model. */
    default boolean isCompatibleRoutedIndexFile(byte[] routingIndexMeta, byte[] indexMeta) {
        return true;
    }
}
