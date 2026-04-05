package com.JJLin.aiagent.client;

public class BrowserOperationException extends Exception {

    private final String errorCode;

    public BrowserOperationException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public BrowserOperationException(String errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public String getErrorCode() {
        return errorCode;
    }
}
