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
import java.io.OutputStream;
import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;

/** Partition-level envelope metadata stored in the vector global-index file. */
public class VectorGlobalIndexFileMeta implements Serializable {

    private static final long serialVersionUID = 1L;

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private static final int VERSION = 1;
    private static final String VERSION_FIELD = "version";
    private static final String SHARD_MODE = "shardMode";

    private final IvfPqShard shardMode;

    private VectorGlobalIndexFileMeta(IvfPqShard shardMode) {
        this.shardMode = shardMode;
        validate();
    }

    public static VectorGlobalIndexFileMeta centroidSharded() {
        return new VectorGlobalIndexFileMeta(IvfPqShard.CENTROID_BASED);
    }

    public byte[] serialize() throws IOException {
        return OBJECT_MAPPER.writeValueAsBytes(toMap());
    }

    public static void writeWithPayload(
            OutputStream out, VectorGlobalIndexFileMeta meta, VectorTrainingModel trainingModel)
            throws IOException {
        meta.writeWithPayload(out, trainingModel);
    }

    public void writeWithPayload(OutputStream out, VectorTrainingModel trainingModel)
            throws IOException {
        byte[] metadata = serialize();
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
        return map;
    }

    public static VectorGlobalIndexFileMeta deserialize(byte[] data) throws IOException {
        JsonNode root = OBJECT_MAPPER.readTree(readMetadataBytes(data));
        validateVersion(root);
        return new VectorGlobalIndexFileMeta(shardMode(root));
    }

    private static byte[] readMetadataBytes(byte[] data) throws IOException {
        if (data.length > 0 && data[0] == '{') {
            return data;
        }
        try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(data))) {
            int metadataLength = in.readInt();
            if (metadataLength <= 0 || metadataLength > data.length - Integer.BYTES) {
                throw new IllegalArgumentException(
                        "Invalid vector global index metadata length: " + metadataLength);
            }
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
    }

    public IvfPqShard shardMode() {
        return shardMode;
    }

    private static void validateVersion(JsonNode root) {
        JsonNode node = root.get(VERSION_FIELD);
        if (node == null) {
            throw new IllegalArgumentException("Vector global index file metadata misses version.");
        }
        int version = node.asInt(-1);
        if (version != VERSION) {
            throw new IllegalArgumentException(
                    "Vector global index file has unsupported version: " + node.asText());
        }
    }

    private static IvfPqShard shardMode(JsonNode root) {
        JsonNode node = root.get(SHARD_MODE);
        if (node == null) {
            throw new IllegalArgumentException("Vector global index file metadata misses shardMode.");
        }
        IvfPqShard shardMode = IvfPqShard.fromValue(node.asText());
        if (shardMode == null) {
            throw new IllegalArgumentException(
                    "Vector global index file has unsupported shardMode: " + node.asText());
        }
        return shardMode;
    }
}
