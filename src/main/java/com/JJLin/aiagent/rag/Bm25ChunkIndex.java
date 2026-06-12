package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.config.KnowledgeProperties;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.cn.smart.SmartChineseAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

@Component
public class Bm25ChunkIndex {

    private final Analyzer analyzer = new SmartChineseAnalyzer();
    private final Directory directory;

    public Bm25ChunkIndex(KnowledgeProperties properties) {
        try {
            this.directory = FSDirectory.open(Path.of(properties.getBm25IndexPath()));
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to open BM25 index.", exception);
        }
    }

    public synchronized void replaceDocument(KnowledgeDocumentRequest request, List<String> chunkIds, List<String> chunks) {
        try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig(analyzer))) {
            writer.deleteDocuments(new Term("documentId", request.getDocumentId()));
            for (int index = 0; index < chunks.size(); index++) {
                Document document = new Document();
                document.add(new StringField("chunkId", chunkIds.get(index), Field.Store.YES));
                document.add(new StringField("documentId", request.getDocumentId(), Field.Store.YES));
                document.add(new StringField("domain", request.getDomain().name(), Field.Store.YES));
                document.add(new StringField("title", request.getTitle(), Field.Store.YES));
                document.add(new StringField("source", request.getSource(), Field.Store.YES));
                document.add(new TextField("text", chunks.get(index), Field.Store.YES));
                document.add(new TextField("tags", tags(request), Field.Store.YES));
                writer.addDocument(document);
            }
            writer.commit();
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to update BM25 index.", exception);
        }
    }

    public synchronized void deleteDocument(String documentId) {
        try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig(analyzer))) {
            writer.deleteDocuments(new Term("documentId", documentId));
            writer.commit();
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to delete document from BM25 index.", exception);
        }
    }

    public List<RagMatch> search(KnowledgeDomain domain, String query, int limit) {
        try {
            if (!DirectoryReader.indexExists(directory)) {
                return List.of();
            }
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                MultiFieldQueryParser parser = new MultiFieldQueryParser(
                        new String[]{"title", "tags", "text"},
                        analyzer,
                        java.util.Map.of("title", 2.0f, "tags", 1.5f, "text", 1.0f));
                BooleanQuery searchQuery = new BooleanQuery.Builder()
                        .add(new TermQuery(new Term("domain", domain.name())), BooleanClause.Occur.FILTER)
                        .add(parser.parse(MultiFieldQueryParser.escape(query)), BooleanClause.Occur.MUST)
                        .build();
                IndexSearcher searcher = new IndexSearcher(reader);
                var topDocs = searcher.search(searchQuery, limit);
                List<RagMatch> results = new ArrayList<>();
                for (var scoreDoc : topDocs.scoreDocs) {
                    Document document = searcher.storedFields().document(scoreDoc.doc);
                    results.add(new RagMatch(document.get("documentId"), document.get("title"),
                            document.get("source"), document.get("text"), scoreDoc.score));
                }
                return results;
            }
        } catch (Exception exception) {
            throw new IllegalStateException("Failed to search BM25 index.", exception);
        }
    }

    private String tags(KnowledgeDocumentRequest request) {
        return request.getTags() == null ? "" : String.join(" ", request.getTags());
    }
}
