import {
  Archive,
  Banknote,
  BrainCircuit,
  CircleDollarSign,
  ClipboardCheck,
  Database,
  Gauge,
  Github,
  GitBranch,
  Landmark,
  Layers3,
  LockKeyhole,
  Mail,
  Network,
  Scale,
  ShieldCheck,
  Sparkles,
  WalletCards,
} from "lucide-react";

import heroBackground from "./assets/finpilot-treasury-hero-v2.png";
import showcaseAnswer from "./assets/showcase/finpilot-grounded-answer.png";
import showcaseInteraction from "./assets/showcase/finpilot-user-interaction.png";
import showcasePlan from "./assets/showcase/finpilot-execution-plan.png";
import showcaseSafety from "./assets/showcase/finpilot-safety-approval.png";
import showcaseStart from "./assets/showcase/finpilot-cli-start.png";

type IconType = typeof ClipboardCheck;

type ShowcaseShot = {
  step: string;
  title: string;
  image: string;
  description: string;
};

type Scenario = {
  title: string;
  description: string;
  Icon: IconType;
};

type ArchitectureItem = {
  title: string;
  description: string;
  Icon: IconType;
};

const navItems = [
  { label: "项目展示", href: "#capabilities" },
  { label: "解决方案", href: "#workflow" },
  { label: "行业场景", href: "#scenarios" },
  { label: "技术架构", href: "#architecture" },
  { label: "关于我们", href: "#about" },
];

const showcaseShots: ShowcaseShot[] = [
  {
    step: "01",
    title: "CLI 启动与会话入口",
    image: showcaseStart,
    description: "FinPilot 本地交互式财资 Agent Shell，展示命令入口、会话状态、上下文窗口和项目元信息。",
  },
  {
    step: "02",
    title: "自然语言财资任务输入",
    image: showcaseInteraction,
    description: "用户直接用业务语言提出工资代发请求，系统进入规划状态并持续展示上下文与运行进度。",
  },
  {
    step: "03",
    title: "多 Agent 执行计划生成",
    image: showcasePlan,
    description: "Planner 将请求拆解为规则查询、余额校验和付款创建等节点，并分派到专业 Agent 并行执行。",
  },
  {
    step: "04",
    title: "高风险操作审批",
    image: showcaseSafety,
    description: "付款与转账类工具调用触发安全审批，明确展示风险原因、工具名称和关键参数。",
  },
  {
    step: "05",
    title: "基于证据的最终回答",
    image: showcaseAnswer,
    description: "系统引用企业制度与银行规则作为依据，说明无法直接执行的原因，并给出需要补充的信息。",
  },
];

const workflowSteps = [
  ["01", "识别业务意图", "理解财资问题、用户角色、目标对象和上下文约束。"],
  ["02", "拆解执行计划", "将请求拆成检索、计算、校验、总结等可执行步骤。"],
  ["03", "调度专业 Agent", "按任务类型调用知识、规则、分析、安全与审计能力。"],
  ["04", "检索知识与规则", "从企业制度、行业知识、案例经验和业务规则中找依据。"],
  ["05", "校验风险合规", "对权限、敏感内容、规则冲突和高风险动作进行前置审查。"],
  ["06", "输出可追溯结论", "给出结构化结果，并记录检索依据、工具调用和决策链路。"],
];

const scenarios: Scenario[] = [
  {
    title: "账户与资产",
    description: "统一查询账户、余额、资产状态和关键变动，形成清晰的财资视图。",
    Icon: WalletCards,
  },
  {
    title: "收付与交易",
    description: "辅助识别回单、交易说明和异常线索，提升处理效率与复核质量。",
    Icon: Banknote,
  },
  {
    title: "资金计划与预测",
    description: "结合历史记录、业务规则和计划口径，辅助判断资金安排与缺口。",
    Icon: Gauge,
  },
  {
    title: "报表与分析",
    description: "把分散数据转化为可解释指标、趋势描述和管理层可读摘要。",
    Icon: ClipboardCheck,
  },
  {
    title: "规则与合规",
    description: "围绕监管、内控、授权和审计要求，发现潜在冲突并提示处置路径。",
    Icon: Scale,
  },
];

const architectureItems: ArchitectureItem[] = [
  {
    title: "LangGraph 编排",
    description: "负责任务路由、节点状态流转和多 Agent 协作流程。",
    Icon: GitBranch,
  },
  {
    title: "RAG 知识检索",
    description: "连接财资制度、行业知识、业务规则和案例资料。",
    Icon: Database,
  },
  {
    title: "记忆与上下文",
    description: "保留会话目标、用户偏好和连续任务中的关键上下文。",
    Icon: BrainCircuit,
  },
  {
    title: "安全护栏",
    description: "对敏感信息、危险操作、越权请求和规则冲突进行拦截。",
    Icon: LockKeyhole,
  },
  {
    title: "工具调用",
    description: "封装检索、计算、导入、诊断和外部系统连接能力。",
    Icon: Network,
  },
  {
    title: "可观测性",
    description: "记录链路、耗时、异常、证据和关键决策过程。",
    Icon: Archive,
  },
  {
    title: "评测与回归",
    description: "用路由、工具、RAG、扎根回答和安全数据集验证能力边界。",
    Icon: Layers3,
  },
];

function SectionBackdrop({ variant = "cool" }: { variant?: "cool" | "warm" | "violet" }) {
  const wash =
    variant === "warm"
      ? "bg-[radial-gradient(circle_at_78%_18%,rgba(216,182,120,0.22),transparent_28%),linear-gradient(135deg,rgba(14,18,24,0.94),rgba(10,12,17,0.82)_52%,rgba(25,20,15,0.9))]"
      : variant === "violet"
        ? "bg-[radial-gradient(circle_at_78%_18%,rgba(139,125,255,0.22),transparent_30%),linear-gradient(135deg,rgba(8,14,22,0.94),rgba(8,10,15,0.84)_55%,rgba(20,16,28,0.9))]"
        : "bg-[radial-gradient(circle_at_78%_18%,rgba(44,181,220,0.2),transparent_30%),linear-gradient(135deg,rgba(7,13,21,0.94),rgba(8,10,15,0.82)_52%,rgba(16,19,24,0.9))]";

  return (
    <>
      <img
        src={heroBackground}
        alt=""
        aria-hidden="true"
        className="absolute inset-0 h-full w-full object-cover object-center opacity-[0.12]"
      />
      <div className={`absolute inset-0 ${wash}`} aria-hidden="true" />
      <div
        className="absolute inset-0 opacity-[0.18] [background-image:linear-gradient(rgba(216,182,120,0.16)_1px,transparent_1px),linear-gradient(90deg,rgba(116,230,255,0.12)_1px,transparent_1px)] [background-size:88px_88px]"
        aria-hidden="true"
      />
    </>
  );
}

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="mx-auto mb-12 max-w-7xl">
      <p className="font-mono text-xs uppercase tracking-[0.18em] text-[#d8b678]">{eyebrow}</p>
      <div className="mt-4 grid gap-5 lg:grid-cols-[0.82fr_1.18fr] lg:items-end">
        <h2 className="font-display text-4xl font-normal leading-tight text-white sm:text-5xl lg:text-6xl">{title}</h2>
        {description ? <p className="max-w-3xl text-base leading-8 text-[#c5cfdd]">{description}</p> : null}
      </div>
    </div>
  );
}

function GlassPanel({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div
      className={`border border-white/12 bg-[#101721]/72 shadow-[0_20px_80px_rgba(0,0,0,0.35)] backdrop-blur-xl ${className}`}
    >
      {children}
    </div>
  );
}

function App() {
  return (
    <main className="min-h-screen overflow-x-hidden bg-[#07090d] text-[#f6f2ea] font-sans">
      <nav
        aria-label="FinPilot 页面导航"
        className="fixed inset-x-0 top-0 z-50 border-b border-white/10 bg-[#05070b]/62 backdrop-blur-xl"
      >
        <div className="mx-auto flex min-h-[76px] max-w-[1760px] items-center justify-between px-5 sm:px-8 lg:px-14">
          <a
            href="#hero"
            className="flex min-h-11 items-center gap-3 rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-[#d8b678]"
          >
            <img
              src="/finpilot-icon.png"
              alt="FinPilot 项目图标"
              className="h-10 w-10 rounded-lg object-cover shadow-[0_0_24px_rgba(45,216,184,0.18)] sm:h-12 sm:w-12"
            />
            <span className="font-display text-3xl text-white sm:text-4xl">FinPilot</span>
            <span className="hidden h-5 w-px bg-white/32 sm:block" aria-hidden="true" />
            <span className="hidden text-sm font-medium tracking-wide text-[#d7d1c5] sm:block">智能财资管理系统</span>
          </a>

          <div className="hidden items-center gap-7 md:flex lg:gap-10">
            {navItems.map((item, index) => (
              <a
                key={item.href}
                href={item.href}
                className={`min-h-11 border-b px-2 py-3 text-sm font-semibold tracking-wide transition duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#d8b678] ${
                  index === 0
                    ? "border-[#8b7dff] text-white"
                    : "border-transparent text-[#d7d1c5] hover:border-white/30 hover:text-white"
                }`}
              >
                {item.label}
              </a>
            ))}
          </div>
        </div>
      </nav>

      <section
        id="hero"
        role="region"
        aria-label="FinPilot 首页首屏"
        className="relative min-h-dvh overflow-hidden px-5 pt-[76px] sm:px-8 lg:px-14"
      >
        <img
          src={heroBackground}
          alt=""
          aria-hidden="true"
          className="absolute inset-0 h-full w-full object-cover object-[58%_center]"
        />
        <div
          className="absolute inset-0 bg-[linear-gradient(90deg,rgba(4,9,14,0.9),rgba(5,10,16,0.74)_32%,rgba(5,9,12,0.22)_66%,rgba(4,6,9,0.35)),linear-gradient(180deg,rgba(5,7,10,0.08),rgba(7,9,12,0.12)_62%,rgba(7,9,12,0.75))]"
          aria-hidden="true"
        />
        <div className="relative z-10 mx-auto flex min-h-[calc(100dvh-76px)] max-w-[1760px] items-center">
          <div
            data-testid="hero-copy"
            className="isolate relative w-full max-w-[900px] pb-[10vh] pt-[8vh] lg:pl-10 xl:pl-16"
          >
            <span
              data-testid="hero-watermark"
              aria-hidden="true"
              className="pointer-events-none absolute -left-2 -top-14 z-0 hidden font-display text-[9rem] leading-none text-white/[0.035] sm:block lg:left-0 lg:text-[12.5rem]"
            >
              FinPilot
            </span>
            <div className="relative z-10">
              <h1 className="font-display text-[5rem] font-normal leading-none text-white drop-shadow-[0_10px_32px_rgba(0,0,0,0.45)] sm:text-[7.5rem] lg:text-[9.25rem]">
                FinPilot
              </h1>
              <p className="mt-10 text-[2rem] font-medium leading-tight text-white sm:text-[2.65rem] lg:text-[3.1rem]">
                企业财资管理的智能中枢
              </p>
              <p className="mt-5 text-lg font-semibold tracking-[0.16em] text-[#d8b678] sm:text-2xl">
                多智能体协同 · 驱动财资智能化
              </p>
              <p className="mt-7 max-w-3xl text-base font-medium leading-8 text-[#d7deea] sm:text-lg">
                融合多智能体协同、行业知识与业务规则，覆盖从账户到交易、从分析到决策的全链路场景，构建安全、可审计、可持续进化的企业财资管理体系。
              </p>
              <span className="mt-9 block h-px w-24 bg-[#d8b678]" aria-hidden="true" />
            </div>
          </div>
        </div>
      </section>

      <section
        id="capabilities"
        role="region"
        aria-label="项目展示"
        className="section-atmosphere relative overflow-hidden px-5 py-24 sm:px-8 lg:px-14"
      >
        <SectionBackdrop variant="warm" />
        <div className="relative z-10">
          <SectionHeading
            eyebrow="Project Showcase"
            title="项目展示"
          />

          <div className="mx-auto grid max-w-7xl gap-5">
            <article className="group overflow-hidden rounded-lg border border-[#d8b678]/34 bg-[#0a0f15]/82 shadow-[0_24px_90px_rgba(0,0,0,0.42)] backdrop-blur-xl transition duration-300 hover:border-[#74e6ff]/45">
              <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
                <div>
                  <span className="font-mono text-xs text-[#74e6ff]">{showcaseShots[0].step}</span>
                  <h3 className="mt-2 text-2xl font-semibold text-white">{showcaseShots[0].title}</h3>
                </div>
                <span className="hidden rounded-full border border-[#d8b678]/30 px-3 py-1 text-xs font-semibold text-[#d8b678] sm:inline-flex">
                  live CLI
                </span>
              </div>
              <div className="bg-[#111]/92 p-3">
                <img
                  src={showcaseShots[0].image}
                  alt={showcaseShots[0].title}
                  className="aspect-[16/7] w-full rounded-md object-contain transition duration-300 group-hover:scale-[1.01]"
                />
              </div>
              <p className="px-5 pb-5 pt-1 text-sm leading-7 text-[#c5cfdd]">{showcaseShots[0].description}</p>
            </article>

            <div className="grid gap-5 lg:grid-cols-2">
              {showcaseShots.slice(1).map(({ step, title, image, description }) => (
                <article
                  key={title}
                  className="group overflow-hidden rounded-lg border border-white/12 bg-[#101721]/76 backdrop-blur-xl transition duration-300 hover:-translate-y-1 hover:border-[#74e6ff]/44 hover:shadow-[0_20px_70px_rgba(0,0,0,0.36)]"
                >
                  <div className="border-b border-white/10 px-5 py-4">
                    <span className="font-mono text-xs text-[#74e6ff]">{step}</span>
                    <h3 className="mt-2 text-xl font-semibold text-white">{title}</h3>
                  </div>
                  <div className="bg-[#111]/92 p-3">
                    <img
                      src={image}
                      alt={title}
                      className="aspect-[16/5] w-full rounded-md object-contain transition duration-300 group-hover:scale-[1.015]"
                    />
                  </div>
                  <p className="px-5 pb-5 pt-1 text-sm leading-7 text-[#c5cfdd]">{description}</p>
                </article>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section
        id="workflow"
        role="region"
        aria-label="解决方案"
        className="section-atmosphere relative overflow-hidden px-5 py-24 sm:px-8 lg:px-14"
      >
        <SectionBackdrop />
        <div className="relative z-10">
          <SectionHeading
            eyebrow="Solution Flow"
            title="解决方案"
          />

          <div className="mx-auto grid max-w-7xl gap-4 md:grid-cols-2 xl:grid-cols-6">
            {workflowSteps.map(([index, title, description]) => (
              <GlassPanel key={title} className="rounded-lg p-6">
                <span className="font-mono text-xs text-[#74e6ff]">{index}</span>
                <h3 className="mt-7 text-xl font-semibold text-white">{title}</h3>
                <p className="mt-4 text-sm leading-7 text-[#c5cfdd]">{description}</p>
              </GlassPanel>
            ))}
          </div>
        </div>
      </section>

      <section
        id="scenarios"
        role="region"
        aria-label="行业场景"
        className="section-atmosphere relative overflow-hidden px-5 py-24 sm:px-8 lg:px-14"
      >
        <SectionBackdrop variant="warm" />
        <div className="relative z-10">
          <SectionHeading
            eyebrow="Treasury Scenarios"
            title="行业场景"
          />

          <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-5">
            {scenarios.map(({ title, description, Icon }) => (
              <article key={title} className="rounded-lg border border-white/12 bg-[#101721]/76 p-6 backdrop-blur-xl">
                <Icon className="h-8 w-8 text-[#d8b678]" strokeWidth={1.6} aria-hidden="true" />
                <h3 className="mt-6 text-xl font-semibold text-white">{title}</h3>
                <p className="mt-4 text-sm leading-7 text-[#c5cfdd]">{description}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section
        id="architecture"
        role="region"
        aria-label="Agent 技术架构"
        className="section-atmosphere relative overflow-hidden px-5 py-24 sm:px-8 lg:px-14"
      >
        <SectionBackdrop variant="violet" />
        <div className="relative z-10">
          <SectionHeading
            eyebrow="Agent Architecture"
            title="Agent 架构"
          />

          <div className="mx-auto max-w-7xl">
            <GlassPanel className="rounded-lg p-5 sm:p-8">
              <div className="grid gap-5">
                <div className="rounded-lg border border-[#d8b678]/28 bg-[#14110d]/82 p-5">
                  <p className="text-sm font-semibold text-[#d8b678]">业务应用层</p>
                  <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                    {["账户与资产", "收付与交易", "资金计划与预测", "报表与分析", "规则与合规"].map((label) => (
                      <span key={label} className="rounded-md border border-white/10 bg-white/[0.06] px-4 py-3 text-center text-sm text-white">
                        {label}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="grid gap-5 lg:grid-cols-[1fr_1.05fr_1fr] lg:items-center">
                  <GlassPanel className="rounded-lg p-5">
                    <p className="text-sm font-semibold text-[#74e6ff]">智能体协同层</p>
                    <div className="mt-5 grid gap-4">
                      {["意图识别", "任务规划", "多智能体协作", "结果校验"].map((label) => (
                        <div key={label} className="flex items-center gap-3 text-sm text-[#d7deea]">
                          <Sparkles className="h-4 w-4 text-[#74e6ff]" strokeWidth={1.8} aria-hidden="true" />
                          {label}
                        </div>
                      ))}
                    </div>
                  </GlassPanel>

                  <div className="relative mx-auto flex aspect-square w-full max-w-[360px] items-center justify-center rounded-full border border-[#74e6ff]/35 bg-[radial-gradient(circle,#17283a_0%,#0d1724_54%,rgba(11,18,28,0.45)_72%,transparent_73%)]">
                    <span className="absolute inset-8 rounded-full border border-[#d8b678]/28" aria-hidden="true" />
                    <span className="absolute inset-16 rounded-full border border-[#8b7dff]/28" aria-hidden="true" />
                    <div className="relative rounded-full border border-white/12 bg-[#101721] px-9 py-8 text-center shadow-[0_0_70px_rgba(68,189,220,0.22)]">
                      <Landmark className="mx-auto h-9 w-9 text-[#d8b678]" strokeWidth={1.5} aria-hidden="true" />
                      <p className="mt-4 text-2xl font-semibold text-white">FinPilot</p>
                      <p className="mt-2 text-sm text-[#bfc7d4]">智能中枢</p>
                    </div>
                  </div>

                  <GlassPanel className="rounded-lg p-5">
                    <p className="text-sm font-semibold text-[#d8b678]">知识与规则层</p>
                    <div className="mt-5 grid gap-4">
                      {["财资知识库", "业务规则库", "监管与合规库", "案例与经验库"].map((label) => (
                        <div key={label} className="flex items-center gap-3 text-sm text-[#d7deea]">
                          <Database className="h-4 w-4 text-[#d8b678]" strokeWidth={1.8} aria-hidden="true" />
                          {label}
                        </div>
                      ))}
                    </div>
                  </GlassPanel>
                </div>

                <div className="rounded-lg border border-[#74e6ff]/24 bg-[#0e1a25]/78 p-5">
                  <p className="text-sm font-semibold text-[#74e6ff]">能力引擎</p>
                  <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                    {architectureItems.map(({ title, description, Icon }) => (
                      <article key={title} className="rounded-md border border-white/10 bg-white/[0.055] p-4">
                        <Icon className="h-6 w-6 text-[#d8b678]" strokeWidth={1.6} aria-hidden="true" />
                        <h3 className="mt-4 text-base font-semibold text-white">{title}</h3>
                        <p className="mt-3 text-sm leading-6 text-[#c5cfdd]">{description}</p>
                      </article>
                    ))}
                  </div>
                </div>

                <div className="rounded-lg border border-white/10 bg-[#0b1017]/86 p-5">
                  <p className="text-sm font-semibold text-[#d8b678]">数据与系统层</p>
                  <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
                    {["核心银行", "ERP", "票据平台", "税务系统", "外部数据", "其他系统"].map((label) => (
                      <span key={label} className="rounded-md border border-white/10 bg-white/[0.05] px-4 py-3 text-center text-sm text-[#d7deea]">
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </GlassPanel>
          </div>
        </div>
      </section>

      <section
        id="about"
        role="region"
        aria-label="关于我们"
        className="section-atmosphere relative overflow-hidden px-5 py-24 sm:px-8 lg:px-14"
      >
        <SectionBackdrop />
        <div className="relative z-10 mx-auto grid max-w-7xl gap-6 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.18em] text-[#d8b678]">Governance</p>
            <h2 className="mt-4 font-display text-4xl font-normal text-white sm:text-5xl lg:text-6xl">可信财资智能</h2>
            <p className="mt-6 max-w-2xl text-base leading-8 text-[#c5cfdd]">
              FinPilot 展示的是企业财资智能化的协同中枢能力：强调清晰链路、可验证依据、风险边界和长期可演进的系统治理。
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <a
                href="https://github.com/JJLin131/FinanceAgent"
                target="_blank"
                rel="noreferrer"
                className="inline-flex min-h-11 items-center gap-2 rounded-md border border-white/12 bg-white/[0.06] px-4 py-2 text-sm font-semibold text-white transition duration-200 hover:border-[#74e6ff]/50 hover:text-[#74e6ff] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#d8b678]"
              >
                <Github className="h-4 w-4" strokeWidth={1.8} aria-hidden="true" />
                GitHub 项目
              </a>
              <a
                href="mailto:yunxiaoli899@gamil.com"
                className="inline-flex min-h-11 items-center gap-2 rounded-md border border-white/12 bg-white/[0.06] px-4 py-2 text-sm font-semibold text-white transition duration-200 hover:border-[#d8b678]/55 hover:text-[#d8b678] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#d8b678]"
              >
                <Mail className="h-4 w-4" strokeWidth={1.8} aria-hidden="true" />
                yunxiaoli899@gamil.com
              </a>
            </div>
          </div>
          <div className="grid gap-4 sm:grid-cols-3">
            {[
              ["来源可追溯", "保留知识、规则、上下文和工具调用依据。", ClipboardCheck],
              ["风险可拦截", "对敏感内容、越权请求和规则冲突做前置审查。", ShieldCheck],
              ["过程可审计", "记录关键节点、校验结论和最终输出链路。", Archive],
            ].map(([title, description, Icon]) => {
              const IconComponent = Icon as IconType;
              return (
                <article key={title as string} className="rounded-lg border border-white/12 bg-[#101721]/76 p-6 backdrop-blur-xl">
                  <IconComponent className="h-8 w-8 text-[#d8b678]" strokeWidth={1.6} aria-hidden="true" />
                  <h3 className="mt-6 text-xl font-semibold text-white">{title as string}</h3>
                  <p className="mt-4 text-sm leading-7 text-[#c5cfdd]">{description as string}</p>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      <footer className="border-t border-white/10 bg-[#05070b] px-5 py-10 text-center text-sm text-[#9aa4b2] sm:px-8">
        <CircleDollarSign className="mx-auto mb-4 h-6 w-6 text-[#d8b678]" strokeWidth={1.6} aria-hidden="true" />
        FinPilot · 企业财资管理的智能中枢
      </footer>
    </main>
  );
}

export default App;
