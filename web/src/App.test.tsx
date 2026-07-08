import { render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import App from "./App";

describe("FinPilot showcase hero", () => {
  it("renders the treasury positioning and primary actions", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: /让财资决策穿透\s*每一层上下文/ })).toBeInTheDocument();
    expect(screen.getAllByText(/Finance Agent Runtime/).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "查看运行链路" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "启动 CLI" })).toBeInTheDocument();
  });

  it("exposes accessible navigation and the interactive reveal layer", () => {
    render(<App />);

    expect(screen.getByRole("navigation", { name: "FinPilot 主导航" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "打开移动导航" })).toBeInTheDocument();
    expect(screen.getByTestId("reveal-layer")).toHaveAttribute("aria-hidden", "true");
  });

  it("keeps the reveal layer hidden until pointer movement paints a mask", async () => {
    const ctx = {
      beginPath: vi.fn(),
      arc: vi.fn(),
      clearRect: vi.fn(),
      createRadialGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
      fill: vi.fn(),
      fillStyle: "",
    };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockReturnValue("data:image/png;base64,mask");

    render(<App />);
    const revealLayer = screen.getByTestId("reveal-layer");

    expect(revealLayer).toHaveStyle({ opacity: "0" });

    const event = new Event("pointermove") as PointerEvent;
    Object.defineProperties(event, {
      clientX: { value: 320 },
      clientY: { value: 240 },
    });
    window.dispatchEvent(event);

    await waitFor(() => {
      expect(revealLayer).toHaveStyle({ opacity: "1" });
      expect(revealLayer).toHaveStyle({ maskImage: "url(data:image/png;base64,mask)" });
    });
  });
});
