package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.config.KnowledgeProperties;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
public class KnowledgeChunker {

    private static final Pattern HEADING = Pattern.compile("^(#{1,6})\\s+(.+?)\\s*$");

    private final KnowledgeProperties properties;

    public KnowledgeChunker(KnowledgeProperties properties) {
        this.properties = properties;
    }

    public List<String> split(String content) {
        String normalized = content == null ? "" : content.strip();
        if (normalized.isEmpty()) {
            return List.of();
        }
        List<Section> sections = parseSections(stripFrontMatter(normalized));
        List<String> chunks = new ArrayList<>();
        for (Section section : sections) {
            String prefix = headingPath(section.headingPath());
            String body = section.body().strip();
            if (body.isEmpty()) {
                continue;
            }
            for (String part : splitOversizedSection(body)) {
                chunks.add(prefix.isEmpty() ? part : prefix + "\n\n" + part);
            }
        }
        return chunks.isEmpty() ? splitOversizedSection(normalized) : chunks;
    }

    private List<Section> parseSections(String content) {
        List<Section> sections = new ArrayList<>();
        List<String> headingStack = new ArrayList<>();
        StringBuilder body = new StringBuilder();
        List<String> currentPath = List.of();

        for (String line : content.split("\\R", -1)) {
            Matcher matcher = HEADING.matcher(line);
            if (matcher.matches()) {
                addSection(sections, currentPath, body);
                int level = matcher.group(1).length();
                while (headingStack.size() >= level) {
                    headingStack.remove(headingStack.size() - 1);
                }
                headingStack.add(matcher.group(2).trim());
                currentPath = List.copyOf(headingStack);
                body = new StringBuilder();
            } else {
                body.append(line).append('\n');
            }
        }
        addSection(sections, currentPath, body);
        return sections;
    }

    private void addSection(List<Section> sections, List<String> path, StringBuilder body) {
        if (!body.toString().isBlank()) {
            sections.add(new Section(path, body.toString()));
        }
    }

    private List<String> splitOversizedSection(String body) {
        int chunkSize = Math.max(100, properties.getChunkSize());
        if (body.length() <= chunkSize) {
            return List.of(body);
        }
        List<String> blocks = markdownBlocks(body);
        List<String> chunks = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (String block : blocks) {
            if (block.length() > chunkSize && !isAtomicMarkdownBlock(block)) {
                flush(chunks, current);
                chunks.addAll(characterSplit(block, chunkSize));
            } else if (current.length() > 0 && current.length() + 2 + block.length() > chunkSize) {
                flush(chunks, current);
                current.append(block);
            } else {
                if (current.length() > 0) {
                    current.append("\n\n");
                }
                current.append(block);
            }
        }
        flush(chunks, current);
        return chunks;
    }

    private List<String> markdownBlocks(String body) {
        List<String> blocks = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean table = false;
        for (String line : body.split("\\R", -1)) {
            boolean tableLine = line.strip().startsWith("|");
            if (!line.isBlank() && (tableLine == table || current.length() == 0)) {
                current.append(line).append('\n');
                table = tableLine;
            } else if (line.isBlank()) {
                flush(blocks, current);
                table = false;
            } else {
                flush(blocks, current);
                current.append(line).append('\n');
                table = tableLine;
            }
        }
        flush(blocks, current);
        return blocks;
    }

    private List<String> characterSplit(String text, int chunkSize) {
        int overlap = Math.max(0, Math.min(properties.getChunkOverlap(), chunkSize / 2));
        List<String> chunks = new ArrayList<>();
        int start = 0;
        while (start < text.length()) {
            int end = Math.min(text.length(), start + chunkSize);
            chunks.add(text.substring(start, end).strip());
            if (end == text.length()) {
                break;
            }
            start = Math.max(start + 1, end - overlap);
        }
        return chunks;
    }

    private String headingPath(List<String> path) {
        return path.isEmpty() ? "" : "章节路径：" + String.join(" > ", path);
    }

    private String stripFrontMatter(String content) {
        if (!content.startsWith("---")) {
            return content;
        }
        int end = content.indexOf("\n---", 3);
        return end < 0 ? content : content.substring(end + 4).strip();
    }

    private boolean isAtomicMarkdownBlock(String block) {
        List<String> lines = block.lines().map(String::strip).filter(line -> !line.isBlank()).toList();
        if (lines.isEmpty()) {
            return false;
        }
        boolean table = lines.stream().allMatch(line -> line.startsWith("|"));
        boolean list = lines.stream().allMatch(line -> line.matches("(-|\\*|\\+)\\s+.*") || line.matches("\\d+[.)]\\s+.*"));
        return table || list;
    }

    private void flush(List<String> values, StringBuilder current) {
        String value = current.toString().strip();
        if (!value.isEmpty()) {
            values.add(value);
        }
        current.setLength(0);
    }

    private record Section(List<String> headingPath, String body) {
    }
}
