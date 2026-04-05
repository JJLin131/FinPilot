package com.JJLin.aiagent.client;

import java.awt.Desktop;
import java.net.URI;

public class DesktopBrowserClient implements BrowserClient {

    @Override
    public void open(URI uri) throws BrowserOperationException {
        if (!Desktop.isDesktopSupported()) {
            throw new BrowserOperationException(
                    "DESKTOP_NOT_SUPPORTED",
                    "Desktop integration is not supported in the current runtime environment.");
        }
        Desktop desktop = Desktop.getDesktop();
        if (!desktop.isSupported(Desktop.Action.BROWSE)) {
            throw new BrowserOperationException(
                    "BROWSE_ACTION_NOT_SUPPORTED",
                    "Desktop browse action is not supported in the current runtime environment.");
        }
        try {
            desktop.browse(uri);
        } catch (SecurityException ex) {
            throw new BrowserOperationException(
                    "BROWSER_BLOCKED_BY_SECURITY_MANAGER",
                    "The operating system or runtime security policy blocked browser launch.",
                    ex);
        } catch (UnsupportedOperationException ex) {
            throw new BrowserOperationException(
                    "BROWSE_ACTION_NOT_SUPPORTED",
                    "Desktop browse action is not supported in the current runtime environment.",
                    ex);
        } catch (IllegalArgumentException ex) {
            throw new BrowserOperationException(
                    "INVALID_URL",
                    "The provided URL is invalid for the system browser.",
                    ex);
        } catch (Exception ex) {
            throw new BrowserOperationException(
                    "BROWSER_OPEN_FAILED",
                    "The system browser could not be opened: " + ex.getMessage(),
                    ex);
        }
    }
}
