package com.JJLin.aiagent.client;

import java.net.URI;

public interface BrowserClient {

    void open(URI uri) throws BrowserOperationException;
}
