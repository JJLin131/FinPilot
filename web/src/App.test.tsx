import { render, screen, within } from "@testing-library/react";
import App from "./App";

describe("FinPilot showcase page", () => {
  it("renders the reference-style hero as a full first screen without bottom capability cards", () => {
    render(<App />);

    const hero = screen.getByRole("region", { name: "FinPilot 首页首屏" });
    const heroCopy = within(hero).getByTestId("hero-copy");
    const watermark = within(hero).getByTestId("hero-watermark");

    expect(within(heroCopy).getByRole("heading", { name: "FinPilot" })).toBeInTheDocument();
    expect(within(heroCopy).getByText("企业财资管理的智能中枢")).toBeInTheDocument();
    expect(within(heroCopy).getByText("多智能体协同 · 驱动财资智能化")).toBeInTheDocument();
    expect(within(heroCopy).getByText(/覆盖从账户到交易、从分析到决策的全链路场景/)).toBeInTheDocument();

    expect(heroCopy).toHaveClass("lg:pl-10");
    expect(heroCopy).toHaveClass("xl:pl-16");
    expect(watermark).toHaveClass("-top-14");
    expect(watermark).toHaveClass("text-white/[0.035]");
    expect(within(hero).queryByText("财资处理与分析")).not.toBeInTheDocument();
    expect(within(hero).queryByText("合规与审计保障")).not.toBeInTheDocument();
    expect(screen.queryByText(/Launch CLI/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("exposes split page sections through the top navigation", () => {
    render(<App />);

    expect(screen.getByRole("navigation", { name: "FinPilot 页面导航" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "FinPilot 项目图标" })).toHaveAttribute(
      "src",
      "/finpilot-icon.png",
    );
    expect(screen.getByRole("link", { name: "产品能力" })).toHaveAttribute("href", "#capabilities");
    expect(screen.getByRole("link", { name: "解决方案" })).toHaveAttribute("href", "#workflow");
    expect(screen.getByRole("link", { name: "行业场景" })).toHaveAttribute("href", "#scenarios");
    expect(screen.getByRole("link", { name: "技术架构" })).toHaveAttribute("href", "#architecture");
    expect(screen.getByRole("link", { name: "关于我们" })).toHaveAttribute("href", "#about");

    expect(screen.getByRole("region", { name: "产品能力" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "解决方案" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "行业场景" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Agent 技术架构" })).toBeInTheDocument();
  });

  it("keeps later sections visually connected to the hero instead of flat black backgrounds", () => {
    render(<App />);

    ["产品能力", "解决方案", "行业场景", "Agent 技术架构", "关于我们"].forEach((label) => {
      const section = screen.getByRole("region", { name: label });
      expect(section).toHaveClass("section-atmosphere");
    });

    ["多智能体协同", "意图识别与理解", "知识与规则驱动", "财资处理与分析", "合规与审计保障"].forEach(
      (label) => {
        expect(screen.getByText(label)).toBeInTheDocument();
      },
    );

    [
      "LangGraph 编排",
      "RAG 知识检索",
      "记忆与上下文",
      "安全护栏",
      "工具调用",
      "可观测性",
      "评测与回归",
    ].forEach((label) => {
      expect(screen.getByText(label)).toBeInTheDocument();
    });
  });

  it("removes internal page-making notes and shows project contacts in about", () => {
    render(<App />);

    expect(screen.queryByText(/首屏只保留参考图式的品牌表达/)).not.toBeInTheDocument();
    expect(screen.queryByText(/用一条可解释链路把用户请求转化为检索/)).not.toBeInTheDocument();
    expect(screen.queryByText(/避免页面只停留在概念层/)).not.toBeInTheDocument();
    expect(screen.queryByText(/参考项目实际架构/)).not.toBeInTheDocument();

    const about = screen.getByRole("region", { name: "关于我们" });
    expect(within(about).getByRole("link", { name: "GitHub 项目" })).toHaveAttribute(
      "href",
      "https://github.com/JJLin131/FinanceAgent",
    );
    expect(within(about).getByRole("link", { name: "yunxiaoli899@gamil.com" })).toHaveAttribute(
      "href",
      "mailto:yunxiaoli899@gamil.com",
    );
  });
});
