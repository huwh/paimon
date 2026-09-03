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

import org.apache.paimon.globalindex.IvfPqShard;

import org.apache.paimon.shade.jackson2.com.fasterxml.jackson.databind.JsonNode;
import org.apache.paimon.shade.jackson2.com.fasterxml.jackson.databind.ObjectMapper;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;

/** Partition-level envelope metadata stored in the vector global-index file. */
public class VectorGlobalIndexFileMeta implements Serializable {

    private static final long serialVersionUID = 1L;

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private static final int LEGACY_VERSION = 1;
    private static final int VERSION = 2;
    private static final String VERSION_FIELD = "version";
    private static final String SHARD_MODE = "shardMode";
    private static final String NLIST = "nlist";
    private static final String MODEL_DIGEST = "modelDigest";
    private static final int MAX_METADATA_LENGTH = 1024 * 1024;

    private final IvfPqShard shardMode;
    private final Integer nlist;
    private final String modelDigest;

    private VectorGlobalIndexFileMeta(IvfPqShard shardMode, Integer nlist, String modelDigest) {
        this.shardMode = shardMode;
        this.nlist = nlist;
        this.modelDigest = modelDigest;
        validate();
    }

    public static VectorGlobalIndexFileMeta centroidSharded() {
        return new VectorGlobalIndexFileMeta(IvfPqShard.CENTROID_BASED, null, null);
    }

    public static VectorGlobalIndexFileMeta centroidSharded(int nlist, String modelDigest) {
        return new VectorGlobalIndexFileMeta(IvfPqShard.CENTROID_BASED, nlist, modelDigest);
    }

    public byte[] serialize() throws IOException {
        if (nlist == null || modelDigest == null) {
            throw new IllegalStateException(
                    "Vector global index metadata requires resolved nlist and modelDigest.");
        }
        return OBJECT_MAPPER.writeValueAsBytes(toMap());
    }

    public static void writeWithPayload(
            OutputStream out, VectorGlobalIndexFileMeta meta, VectorTrainingModel trainingModel)
            throws IOException {
        meta.writeWithPayload(out, trainingModel);
    }

    public void writeWithPayload(OutputStream out, VectorTrainingModel trainingModel)
            throws IOException {
        VectorGlobalIndexFileMeta resolved =
                centroidSharded(trainingModel.centroids().nlist(), trainingModel.modelDigest());
        if (nlist != null && !nlist.equals(resolved.nlist)) {
            throw new IllegalArgumentException(
                    "Vector global index metadata nlist does not match the training model.");
        }
        if (modelDigest != null && !modelDigest.equals(resolved.modelDigest)) {
            throw new IllegalArgumentException(
                    "Vector global index metadata digest does not match the training model.");
        }
        byte[] metadata = resolved.serialize();
        DataOutputStream dataOutputStream = new DataOutputStream(out);
        dataOutputStream.writeInt(metadata.length);
        dataOutputStream.write(metadata);
        trainingModel.serializeNativeModelPayloadTo(dataOutputStream);
        dataOutputStream.flush();
    }

    private Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put(VERSION_FIELD, VERSION);
        map.put(SHARD_MODE, shardMode.optionValue());
        if (nlist != null) {
            map.put(NLIST, nlist);
        }
        if (modelDigest != null) {
            map.put(MODEL_DIGEST, modelDigest);
        }
        return map;
    }

    public static VectorGlobalIndexFileMeta deserialize(byte[] data) throws IOException {
        JsonNode root = OBJECT_MAPPER.readTree(readMetadataBytes(data));
        return fromJson(root);
    }

    /** Reads only the bounded envelope header, leaving {@code in} positioned at the payload. */
    public static ArtifactHeader readArtifactHeader(InputStream in, long artifactSize)
            throws IOException {
        if (artifactSize < Integer.BYTES) {
            throw new IllegalArgumentException(
                    "Invalid vector global index artifact size: " + artifactSize);
        }
        DataInputStream dataInput =
                in instanceof DataInputStream ? (DataInputStream) in : new DataInputStream(in);
        int metadataLength = dataInput.readInt();
        validateMetadataLength(metadataLength, artifactSize - Integer.BYTES);
        byte[] metadata = new byte[metadataLength];
        dataInput.readFully(metadata);
        VectorGlobalIndexFileMeta meta = fromJson(OBJECT_MAPPER.readTree(metadata));
        return new ArtifactHeader(meta, artifactSize - Integer.BYTES - metadataLength);
    }

    private static byte[] readMetadataBytes(byte[] data) throws IOException {
        if (data.length > 0 && data[0] == '{') {
            return data;
        }
        try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(data))) {
            int metadataLength = in.readInt();
            validateMetadataLength(metadataLength, data.length - Integer.BYTES);
            byte[] metadata = new byte[metadataLength];
            in.readFully(metadata);
            return metadata;
        }
    }

    private void validate() {
        if (shardMode != IvfPqShard.CENTROID_BASED) {
            throw new IllegalArgumentException(
                    "Vector global index file requires shardMode="
                            + IvfPqShard.CENTROID_BASED.optionValue()
                            + ".");
        }
        if ((nlist == null) != (modelDigest == null)) {
            throw new IllegalArgumentException(
                    "Vector global index metadata must contain both nlist and modelDigest.");
        }
        if (nlist != null && nlist <= 0) {
            throw new IllegalArgumentException(
                    "Vector global index metadata requires positive nlist: " + nlist);
        }
        if (modelDigest != null) {
            VectorModelDigest.validate(modelDigest);
        }
    }

    public IvfPqShard shardMode() {
        return shardMode;
    }

    public int nlist() {
        if (nlist == null) {
            throw new IllegalStateException("Vector global index metadata misses nlist.");
        }
        return nlist;
    }

    public String modelDigest() {
        if (modelDigest == null) {
            throw new IllegalStateException("Vector global index metadata misses modelDigest.");
        }
        return modelDigest;
    }

    public boolean hasModelIdentity() {
        return nlist != null && modelDigest != null;
    }

    private static VectorGlobalIndexFileMeta fromJson(JsonNode root) {
        int version = validateVersion(root);
        JsonNode nlist = root.get(NLIST);
        JsonNode modelDigest = root.get(MODEL_DIGEST);
        if (version == VERSION && (nlist == null || modelDigest == null)) {
            throw new IllegalArgumentException(
                    "Vector global index file metadata misses nlist or modelDigest.");
        }
        return new VectorGlobalIndexFileMeta(
                shardMode(root),
                nlist == null ? null : nlist.asInt(-1),
                modelDigest == null ? null : modelDigest.asText());
    }

    private static void validateMetadataLength(int metadataLength, long remainingBytes) {
        if (metadataLength <= 0
                || metadataLength > MAX_METADATA_LENGTH
                || metadataLength > remainingBytes) {
            throw new IllegalArgumentException(
                    "Invalid vector global index metadata length: " + metadataLength);
        }
    }

    /** Parsed artifact metadata and remaining native payload size. */
    public static class ArtifactHeader {

        private final VectorGlobalIndexFileMeta metadata;
        private final long payloadSize;

        private ArtifactHeader(VectorGlobalIndexFileMeta metadata, long payloadSize) {
            this.metadata = metadata;
            this.payloadSize = payloadSize;
        }

        public VectorGlobalIndexFileMeta metadata() {
            return metadata;
        }

        public long payloadSize() {
            return payloadSize;
        }
    }

    private static int validateVersion(JsonNode root) {
        JsonNode node = root.get(VERSION_FIELD);
        if (node == null) {
            throw new IllegalArgumentException("Vector global index file metadata misses version.");
        }
        int version = node.asInt(-1);
        if (version != LEGACY_VERSION && version != VERSION) {
            throw new IllegalArgumentException(
                    "Vector global index file has unsupported version: " + node.asText());
        }
        return version;
    }

    private static IvfPqShard shardMode(JsonNode root) {
        JsonNode node = root.get(SHARD_MODE);
        if (node == null) {
            throw new IllegalArgumentException(
                    "Vector global index file metadata misses shardMode.");
        }
        IvfPqShard shardMode = IvfPqShard.fromValue(node.asText());
        if (shardMode == null) {
            throw new IllegalArgumentException(
                    "Vector global index file has unsupported shardMode: " + node.asText());
        }
        return shardMode;
    }
}
