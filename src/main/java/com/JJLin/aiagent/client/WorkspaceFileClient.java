package com.JJLin.aiagent.client;

import com.JJLin.aiagent.config.CapabilityProperties;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class WorkspaceFileClient {

    private final Path workspaceRoot;
    private final int maxFileReadChars;

    public WorkspaceFileClient(CapabilityProperties properties) {
        this.workspaceRoot = Path.of(properties.getWorkspaceRoot()).toAbsolutePath().normalize();
        this.maxFileReadChars = properties.getMaxFileReadChars() == null ? 20000 : properties.getMaxFileReadChars();
    }

    public List<String> listFiles(String relativePath, boolean recursive, int maxEntries) throws IOException {
        Path target = resolve(relativePath);
        if (!Files.exists(target)) {
            throw new IllegalArgumentException("Path does not exist: " + relativePath);
        }
        if (!Files.isDirectory(target)) {
            throw new IllegalArgumentException("Path is not a directory: " + relativePath);
        }

        int effectiveMaxEntries = maxEntries <= 0 ? 100 : maxEntries;
        try (Stream<Path> stream = recursive ? Files.walk(target) : Files.list(target)) {
            return stream
                    .filter(path -> !path.equals(target))
                    .sorted(Comparator.naturalOrder())
                    .limit(effectiveMaxEntries)
                    .map(path -> workspaceRoot.relativize(path.toAbsolutePath().normalize()).toString())
                    .toList();
        }
    }

    public String readFile(String relativePath) throws IOException {
        Path target = resolve(relativePath);
        if (!Files.exists(target)) {
            throw new IllegalArgumentException("File does not exist: " + relativePath);
        }
        if (!Files.isRegularFile(target)) {
            throw new IllegalArgumentException("Path is not a file: " + relativePath);
        }
        String content = Files.readString(target, StandardCharsets.UTF_8);
        if (content.length() <= maxFileReadChars) {
            return content;
        }
        return content.substring(0, maxFileReadChars);
    }

    public void writeFile(String relativePath, String content, boolean append) throws IOException {
        Path target = resolve(relativePath);
        Path parent = target.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        if (append) {
            Files.writeString(target, content, StandardCharsets.UTF_8,
                    java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
            return;
        }
        Files.writeString(target, content, StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.TRUNCATE_EXISTING);
    }

    public Path workspaceRoot() {
        return workspaceRoot;
    }

    private Path resolve(String relativePath) {
        String safeRelativePath = relativePath == null || relativePath.isBlank() ? "." : relativePath;
        Path resolved = workspaceRoot.resolve(safeRelativePath).normalize().toAbsolutePath();
        if (!resolved.startsWith(workspaceRoot)) {
            throw new IllegalArgumentException("Path escapes workspace root: " + safeRelativePath);
        }
        return resolved;
    }
}
