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
import org.apache.paimon.globalindex.GlobalIndexWriter;
import org.apache.paimon.globalindex.IndexFileKind;
import org.apache.paimon.globalindex.IvfPqShard;
import org.apache.paimon.globalindex.VectorGlobalIndexer;
import org.apache.paimon.globalindex.io.GlobalIndexFileReader;
import org.apache.paimon.globalindex.io.GlobalIndexFileWriter;
import org.apache.paimon.types.DataType;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutorService;

/** Native vector global indexer backed by paimon-vector-index-java. */
public class NativeVectorGlobalIndexer implements VectorGlobalIndexer {

    static final String DEFAULT_METRIC = "inner_product";

    private final DataType fieldType;
    private final Map<String, String> options;
    private final String identifier;
    private final double trainSampleRatio;

    public NativeVectorGlobalIndexer(
            DataType fieldType, Map<String, String> options, String identifier) {
        this(
                fieldType,
                options,
                identifier,
                NativeVectorGlobalIndexerFactory.DEFAULT_TRAIN_SAMPLE_RATIO);
    }

    public NativeVectorGlobalIndexer(
            DataType fieldType,
            Map<String, String> options,
            String identifier,
            double trainSampleRatio) {
        this.fieldType = fieldType;
        this.options = Objects.requireNonNull(options, "options must not be null");
        this.identifier = Objects.requireNonNull(identifier, "identifier must not be null");
        if (Double.isNaN(trainSampleRatio)
                || Double.isInfinite(trainSampleRatio)
                || trainSampleRatio <= 0
                || trainSampleRatio > 1) {
            throw new IllegalArgumentException(
                    "trainSampleRatio must be greater than 0 and less than or equal to 1: "
                            + trainSampleRatio);
        }
        this.trainSampleRatio = trainSampleRatio;
    }

    DataType fieldType() {
        return fieldType;
    }

    @Override
    public GlobalIndexWriter createWriter(GlobalIndexFileWriter fileWriter) {
        return new NativeVectorGlobalIndexWriter(
                fileWriter, fieldType, options, identifier, trainSampleRatio);
    }

    @Override
    public GlobalIndexReader createReader(
            GlobalIndexFileReader fileReader,
            List<GlobalIndexIOMeta> files,
            long totalRowCount,
            ExecutorService executor) {
        CentroidRoutedFiles centroidRoutedFiles = CentroidRoutedFiles.tryCreate(fileReader, files);
        if (centroidRoutedFiles != null) {
            return createCentroidRoutedReader(fileReader, centroidRoutedFiles, executor);
        }
        Map<Integer, GlobalIndexIOMeta> centroidDataShardFiles;
        try {
            centroidDataShardFiles = CentroidRoutedFiles.tryCreateCentroidDataShardFiles(files);
        } catch (IOException e) {
            throw new RuntimeException("Failed to parse vector index shard metadata", e);
        }
        if (!centroidDataShardFiles.isEmpty()) {
            return new NativeCentroidRoutedIndexReader(
                    fileReader, null, centroidDataShardFiles, fieldType, executor, options);
        }
        return new NativeVectorGlobalIndexReader(fileReader, files, fieldType, executor);
    }

    private GlobalIndexReader createCentroidRoutedReader(
            GlobalIndexFileReader fileReader, CentroidRoutedFiles files, ExecutorService executor) {
        try {
            return new NativeCentroidRoutedIndexReader(
                    fileReader,
                    new NativeVectorCentroidRouter(fileReader, files.globalIndexFile()),
                    files.centroidToFile(),
                    fieldType,
                    executor,
                    options);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create centroid-routed vector index reader", e);
        }
    }

    @Override
    public String metric() {
        return options.getOrDefault("metric", DEFAULT_METRIC);
    }

    @Override
    public boolean isRoutingGlobalIndexFile(IndexFileKind fileKind, byte[] indexMeta) {
        if (fileKind != IndexFileKind.ROUTING_MODEL || indexMeta == null) {
            return false;
        }
        try {
            return VectorIndexMeta.deserialize(indexMeta).shardMode() == IvfPqShard.CENTROID_BASED;
        } catch (IOException | IllegalArgumentException e) {
            return false;
        }
    }

    @Override
    public Set<Integer> routeCentroids(
            GlobalIndexFileReader fileReader,
            GlobalIndexIOMeta globalIndexFile,
            float[][] queryVectors,
            int limit,
            Map<String, String> options) {
        try {
            VectorIndexMeta routingMeta = VectorIndexMeta.deserialize(globalIndexFile.metadata());
            if (!routingMeta.hasModelIdentity()) {
                return java.util.Collections.emptySet();
            }
            Set<Integer> routed = new LinkedHashSet<>();
            try (VectorTrainingModel model =
                    NativeVectorTrainingModels.load(fileReader, globalIndexFile)) {
                if (routingMeta.nlist() == null
                        || routingMeta.nlist() != model.centroids().nlist()
                        || routingMeta.modelDigest() == null
                        || !routingMeta.modelDigest().equals(model.modelDigest())) {
                    throw new IOException(
                            "Routing manifest metadata does not match the vector model artifact: "
                                    + globalIndexFile.filePath().getName());
                }
                int nprobe = nprobe(options, limit);
                for (float[] queryVector : queryVectors) {
                    for (int centroid : model.centroids().nearestCentroids(queryVector, nprobe)) {
                        routed.add(centroid);
                    }
                }
            }
            return routed;
        } catch (IOException e) {
            throw new RuntimeException("Failed to route vector centroid ids.", e);
        }
    }

    @Override
    public boolean acceptsRoutedIndexFile(byte[] indexMeta, Set<Integer> routedCentroids) {
        return acceptsRoutedIndexFile(null, indexMeta, routedCentroids);
    }

    @Override
    public boolean acceptsRoutedIndexFile(
            byte[] routingIndexMeta, byte[] indexMeta, Set<Integer> routedCentroids) {
        if (indexMeta == null || routedCentroids.isEmpty()) {
            return false;
        }
        try {
            VectorIndexMeta meta = VectorIndexMeta.deserialize(indexMeta);
            if (!meta.isCentroidShard()
                    || meta.rowIdEncoding() != VectorIndexMeta.RowIdEncoding.ABSOLUTE_ROW_ID
                    || meta.centroid() == null) {
                return false;
            }
            if (routingIndexMeta != null
                    && !isCompatibleRoutedIndexFile(routingIndexMeta, indexMeta)) {
                return false;
            }
            return meta.modelDigest() != null && routedCentroids.contains(meta.centroid());
        } catch (IOException | IllegalArgumentException e) {
            return false;
        }
    }

    @Override
    public boolean isCompatibleRoutedIndexFile(byte[] routingIndexMeta, byte[] indexMeta) {
        if (routingIndexMeta == null || indexMeta == null) {
            return false;
        }
        try {
            VectorIndexMeta routingMeta = VectorIndexMeta.deserialize(routingIndexMeta);
            VectorIndexMeta shardMeta = VectorIndexMeta.deserialize(indexMeta);
            return routingMeta.shardMode() == IvfPqShard.CENTROID_BASED
                    && routingMeta.modelDigest() != null
                    && shardMeta.isCentroidShard()
                    && routingMeta.modelDigest().equals(shardMeta.modelDigest());
        } catch (IOException | IllegalArgumentException e) {
            return false;
        }
    }

    private static int nprobe(Map<String, String> options, int limit) {
        String value = options.get("ivf.nprobe");
        if (value == null) {
            return limit;
        }
        int nprobe = Integer.parseInt(value);
        if (nprobe <= 0) {
            throw new IllegalArgumentException("Invalid value for 'ivf.nprobe': " + value);
        }
        return nprobe;
    }

    static class CentroidRoutedFiles {

        private final GlobalIndexIOMeta globalIndexFile;
        private final Map<Integer, GlobalIndexIOMeta> centroidToFile;

        private CentroidRoutedFiles(
                GlobalIndexIOMeta globalIndexFile, Map<Integer, GlobalIndexIOMeta> centroidToFile) {
            this.globalIndexFile = globalIndexFile;
            this.centroidToFile = centroidToFile;
        }

        GlobalIndexIOMeta globalIndexFile() {
            return globalIndexFile;
        }

        Map<Integer, GlobalIndexIOMeta> centroidToFile() {
            return centroidToFile;
        }

        static CentroidRoutedFiles tryCreate(
                GlobalIndexFileReader fileReader, List<GlobalIndexIOMeta> files) {
            try {
                if (files.isEmpty()) {
                    return null;
                }

                GlobalIndexIOMeta globalIndexFile = null;
                for (GlobalIndexIOMeta file : files) {
                    if (file.fileKind() != IndexFileKind.ROUTING_MODEL) {
                        continue;
                    }
                    VectorIndexMeta fileMeta = parseVectorIndexMeta(file.metadata());
                    if (fileMeta == null) {
                        throw new IllegalArgumentException(
                                "Routing model index file requires vector index metadata: "
                                        + file.filePath().getName());
                    }
                    if (fileMeta.shardMode() != IvfPqShard.CENTROID_BASED) {
                        throw new IllegalArgumentException(
                                "Centroid-routed vector index requires shardMode="
                                        + IvfPqShard.CENTROID_BASED.optionValue()
                                        + ".");
                    }
                    if (globalIndexFile != null) {
                        throw new IllegalArgumentException(
                                "Centroid-routed vector index currently supports only one "
                                        + "global index file per partition.");
                    }
                    globalIndexFile = file;
                }
                if (globalIndexFile == null) {
                    return null;
                }

                VectorIndexMeta routingMeta = parseVectorIndexMeta(globalIndexFile.metadata());
                if (!routingMeta.hasModelIdentity()) {
                    return null;
                }
                VectorGlobalIndexFileMeta globalMeta;
                try (org.apache.paimon.fs.SeekableInputStream in =
                        fileReader.getInputStream(globalIndexFile)) {
                    globalMeta =
                            VectorGlobalIndexFileMeta.readArtifactHeader(
                                            in, globalIndexFile.fileSize())
                                    .metadata();
                }
                if (globalMeta.shardMode() != IvfPqShard.CENTROID_BASED) {
                    throw new IllegalArgumentException(
                            "Vector global index file contains shardMode="
                                    + globalMeta.shardMode().optionValue()
                                    + ".");
                }
                if (routingMeta.nlist() == null
                        || routingMeta.nlist() != globalMeta.nlist()
                        || routingMeta.modelDigest() == null
                        || !routingMeta.modelDigest().equals(globalMeta.modelDigest())) {
                    throw new IllegalArgumentException(
                            "Routing manifest metadata does not match vector model artifact: "
                                    + globalIndexFile.filePath().getName());
                }
                Map<Integer, GlobalIndexIOMeta> centroidToFile =
                        tryCreateCentroidDataShardFiles(files);
                for (Map.Entry<Integer, GlobalIndexIOMeta> shard : centroidToFile.entrySet()) {
                    VectorIndexMeta shardMeta = parseVectorIndexMeta(shard.getValue().metadata());
                    if (!globalMeta.modelDigest().equals(shardMeta.modelDigest())) {
                        throw new IllegalArgumentException(
                                "Centroid shard model digest does not match routing model: file="
                                        + shard.getValue().filePath().getName()
                                        + ", centroid="
                                        + shard.getKey());
                    }
                    if (shard.getKey() < 0 || shard.getKey() >= globalMeta.nlist()) {
                        throw new IllegalArgumentException(
                                "Centroid shard id is outside routing model nlist: centroid="
                                        + shard.getKey()
                                        + ", nlist="
                                        + globalMeta.nlist());
                    }
                }
                for (GlobalIndexIOMeta file : files) {
                    String fileName = file.filePath().getName();
                    if (file == globalIndexFile
                            || file.filePath().equals(globalIndexFile.filePath())) {
                        continue;
                    }
                    if (!containsFile(centroidToFile, file)) {
                        throw new IllegalArgumentException(
                                "Centroid-routed vector index currently supports only centroid "
                                        + "data shards in one partition. Unknown vector index file: "
                                        + fileName);
                    }
                }
                return new CentroidRoutedFiles(globalIndexFile, centroidToFile);
            } catch (IOException e) {
                throw new RuntimeException("Failed to parse vector index metadata", e);
            }
        }

        static Map<Integer, GlobalIndexIOMeta> tryCreateCentroidDataShardFiles(
                List<GlobalIndexIOMeta> files) throws IOException {
            Map<Integer, GlobalIndexIOMeta> centroidToFile = new LinkedHashMap<>();
            String modelDigest = null;
            Boolean identifiedMetadata = null;
            for (GlobalIndexIOMeta file : files) {
                if (file.fileKind() == IndexFileKind.ROUTING_MODEL) {
                    continue;
                }
                VectorIndexMeta fileMeta = parseVectorIndexMeta(file.metadata());
                VectorIndexMeta shardMeta =
                        parseCentroidShardMeta(fileMeta, file.filePath().getName());
                if (shardMeta == null) {
                    continue;
                }
                boolean identified = shardMeta.modelDigest() != null;
                if (identifiedMetadata == null) {
                    identifiedMetadata = identified;
                } else if (identifiedMetadata != identified) {
                    throw new IllegalArgumentException(
                            "Centroid data shards mix legacy and identified models: file="
                                    + file.filePath().getName());
                }
                if (identified && modelDigest == null) {
                    modelDigest = shardMeta.modelDigest();
                } else if (identified && !modelDigest.equals(shardMeta.modelDigest())) {
                    throw new IllegalArgumentException(
                            "Centroid data shards use different model digests: file="
                                    + file.filePath().getName());
                }
                GlobalIndexIOMeta previous = centroidToFile.put(shardMeta.centroid(), file);
                if (previous != null) {
                    throw new IllegalArgumentException(
                            "Duplicate centroid data shard for centroid=" + shardMeta.centroid());
                }
            }
            return centroidToFile;
        }

        private static boolean containsFile(
                Map<Integer, GlobalIndexIOMeta> centroidToFile, GlobalIndexIOMeta file) {
            for (GlobalIndexIOMeta dataShard : centroidToFile.values()) {
                if (dataShard == file || dataShard.filePath().equals(file.filePath())) {
                    return true;
                }
            }
            return false;
        }

        private static VectorIndexMeta parseCentroidShardMeta(
                VectorIndexMeta meta, String fileName) {
            if (meta == null) {
                return null;
            }
            try {
                if (!meta.isCentroidShard()) {
                    return null;
                }
                if (meta.centroid() == null) {
                    throw new IllegalArgumentException(
                            "Centroid shard index file requires centroid: " + fileName);
                }
                if (meta.rowIdEncoding() != VectorIndexMeta.RowIdEncoding.ABSOLUTE_ROW_ID) {
                    throw new IllegalArgumentException(
                            "Centroid shard index file requires ABSOLUTE_ROW_ID encoding: "
                                    + fileName);
                }
                return meta;
            } catch (IllegalArgumentException e) {
                throw e;
            }
        }

        private static VectorIndexMeta parseVectorIndexMeta(byte[] indexMeta) throws IOException {
            if (indexMeta == null) {
                return null;
            }
            return VectorIndexMeta.deserialize(indexMeta);
        }
    }
}
