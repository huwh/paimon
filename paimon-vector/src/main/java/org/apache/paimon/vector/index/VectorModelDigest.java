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

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** Digest utilities for identifying a serialized vector training model. */
final class VectorModelDigest {

    static final String PREFIX = "sha256:";
    private static final char[] HEX = "0123456789abcdef".toCharArray();

    private VectorModelDigest() {}

    static String sha256(byte[] payload) {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available.", e);
        }
        byte[] hash = digest.digest(payload);
        char[] hex = new char[hash.length * 2];
        for (int i = 0; i < hash.length; i++) {
            int value = hash[i] & 0xff;
            hex[i * 2] = HEX[value >>> 4];
            hex[i * 2 + 1] = HEX[value & 0x0f];
        }
        return PREFIX + new String(hex);
    }

    static void validate(String digest) {
        if (digest == null
                || digest.length() != PREFIX.length() + 64
                || !digest.startsWith(PREFIX)) {
            throw new IllegalArgumentException(
                    "Vector training model digest must use the format sha256:<64 lowercase hex characters>.");
        }
        for (int i = PREFIX.length(); i < digest.length(); i++) {
            char c = digest.charAt(i);
            if ((c < '0' || c > '9') && (c < 'a' || c > 'f')) {
                throw new IllegalArgumentException(
                        "Vector training model digest must use the format sha256:<64 lowercase hex characters>.");
            }
        }
    }
}
