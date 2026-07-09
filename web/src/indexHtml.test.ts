import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

describe("HTML shell", () => {
  it("uses the FinPilot project icon as the favicon", () => {
    const html = readFileSync(resolve(process.cwd(), "index.html"), "utf8");

    expect(html).toContain('href="/finpilot-icon.png"');
    expect(html).not.toContain("vite.svg");
    expect(existsSync(resolve(process.cwd(), "public/finpilot-icon.png"))).toBe(true);
  });
});
