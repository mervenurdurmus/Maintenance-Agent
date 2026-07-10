import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  BarChart3,
  Home,
  ImagePlus,
  MessageSquare,
  Plus,
  Send,
  Settings,
  Trash2,
  X,
} from "lucide-react";
import {
  createConversation,
  deleteConversation,
  getAppSettings,
  getEvaluationStatus,
  getConversationMessages,
  getConversationToolCalls,
  listConversations,
  resolveAssetUrl,
  sendMessage,
  updateChatLlmSettings,
} from "./api/client";
import { FileUpload } from "./components/FileUpload";
import { MarkdownMessage } from "./components/MarkdownMessage";
import type {
  ChatResponse,
  AppSettings,
  ConversationInfo,
  EvaluationStatus,
  HistoryToolCall,
  ImageAttachment,
} from "./types/api";

type ChatItem = {
  role: "user" | "assistant";
  content: string;
  turnId?: string | null;
  metadata?: ChatResponse;
  attachments?: ImageAttachment[];
};

type TurnToolSummary = {
  turnId: string;
  userMessage: string;
  toolCalls: HistoryToolCall[];
};

type AppPage = "home" | "chat" | "ragas";
type RagasProvider = "groq" | "openrouter";
type ChatSidebarTab = "history" | "settings";

function formatDateTime(value: string) {
  const date = parseDateTime(value);
  const now = new Date();

  if (Number.isNaN(date.getTime())) return value;

  const dateKey = date.toDateString();
  const todayKey = now.toDateString();
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);

  const time = date.toLocaleTimeString("tr-TR", {
    hour: "2-digit",
    minute: "2-digit",
  });

  if (dateKey === todayKey) {
    return `Bugün ${time}`;
  }

  if (dateKey === yesterday.toDateString()) {
    return `Dün ${time}`;
  }

  const options: Intl.DateTimeFormatOptions = {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  };

  if (date.getFullYear() !== now.getFullYear()) {
    options.year = "numeric";
  }

  return date.toLocaleString("tr-TR", options);
}

function parseDateTime(value: string) {
  const normalized = value.includes("T") ? value : value.replace(" ", "T");
  const hasTimezone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(normalized);

  return new Date(hasTimezone ? normalized : `${normalized}Z`);
}

export function App() {
  const [activePage, setActivePage] = useState<AppPage>("home");
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState<ChatItem[]>([]);
  const [conversations, setConversations] = useState<ConversationInfo[]>([]);
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
  const [chatSidebarTab, setChatSidebarTab] = useState<ChatSidebarTab>("history");
  const [evaluationStatus, setEvaluationStatus] = useState<EvaluationStatus | null>(null);
  const [ragasProvider, setRagasProvider] = useState<RagasProvider>("groq");
  const [ragasModel, setRagasModel] = useState("openai/gpt-oss-20b");
  const [chatProvider, setChatProvider] = useState<RagasProvider>("groq");
  const [chatModel, setChatModel] = useState("openai/gpt-oss-120b");
  const [settingsMessage, setSettingsMessage] = useState("");
  const [isEvaluationLoading, setIsEvaluationLoading] = useState(false);
  const [toolCalls, setToolCalls] = useState<HistoryToolCall[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [selectedResponse, setSelectedResponse] = useState<ChatResponse | null>(null);
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [selectedImagePreview, setSelectedImagePreview] = useState<string | null>(null);
  const [viewedImage, setViewedImage] = useState<ImageAttachment | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const chatWindowRef = useRef<HTMLElement | null>(null);
  const messageInputRef = useRef<HTMLInputElement | null>(null);
  const initializedRef = useRef(false);
  const selectedToolCalls = selectedTurnId
    ? toolCalls.filter((tool) => tool.turn_id === selectedTurnId)
    : [];
  const latestSources = [...chat]
    .reverse()
    .find((item) => item.metadata?.sources.length)?.metadata?.sources ?? [];
  const displayedSources = selectedResponse?.sources.length
    ? selectedResponse.sources
    : latestSources;
  const turnToolSummaries: TurnToolSummary[] = chat.flatMap((item, index) => {
    if (item.role !== "assistant" || !item.turnId) return [];

    const previousUserMessage = [...chat.slice(0, index)]
      .reverse()
      .find(
        (previousItem) =>
          previousItem.role === "user" &&
          (previousItem.turnId === item.turnId || !previousItem.turnId),
      );

    return [
      {
        turnId: item.turnId,
        userMessage: previousUserMessage?.content ?? "Kullanıcı mesajı bulunamadı",
        toolCalls: toolCalls.filter((tool) => tool.turn_id === item.turnId),
      },
    ];
  });

  useEffect(() => {
    if (initializedRef.current) return;
    initializedRef.current = true;

    async function initializeConversations() {
      try {
        const items = await listConversations();
        setConversations(items);

        if (items.length > 0) {
          await openConversation(items[0].conversation_id);
        } else {
          await handleNewConversation();
        }
      } catch {
        setChat([
          {
            role: "assistant",
            content: "Sohbet geçmişi yüklenemedi. Backend bağlantısını kontrol et.",
          },
        ]);
      }
    }

    void initializeConversations();
  }, []);

  useEffect(() => {
    const chatWindow = chatWindowRef.current;
    if (!chatWindow) return;

    chatWindow.scrollTop = chatWindow.scrollHeight;
  }, [chat, isBusy]);

  useEffect(() => {
    return () => {
      if (selectedImagePreview) {
        URL.revokeObjectURL(selectedImagePreview);
      }
    };
  }, [selectedImagePreview]);

  useEffect(() => {
    if (activePage !== "ragas") return;

    async function loadEvaluationStatus() {
      setIsEvaluationLoading(true);
      try {
        const status = await getEvaluationStatus();
        setEvaluationStatus(status);
      } finally {
        setIsEvaluationLoading(false);
      }
    }

    void loadEvaluationStatus();
  }, [activePage]);

  useEffect(() => {
    if (activePage !== "chat" || chatSidebarTab !== "settings" || appSettings) return;

    async function loadAppSettings() {
      try {
        const settings = await getAppSettings();
        setAppSettings(settings);
        const defaultProvider = settings.ragas_llm.default_provider;
        const defaultProviderInfo = settings.ragas_llm.providers.find(
          (provider) => provider.id === defaultProvider,
        );
        setRagasProvider(defaultProvider);
        setRagasModel(defaultProviderInfo?.model ?? settings.chat_model);
        setChatProvider(settings.chat_llm.active_provider);
        setChatModel(settings.chat_llm.active_model);
      } catch {
        setAppSettings(null);
      }
    }

    void loadAppSettings();
  }, [activePage, chatSidebarTab, appSettings]);

  async function refreshConversations() {
    const items = await listConversations();
    setConversations(items);
  }

  async function openConversation(id: string) {
    setIsBusy(true);
    try {
      const [messages, savedToolCalls] = await Promise.all([
        getConversationMessages(id),
        getConversationToolCalls(id),
      ]);
      const loadedChat = messages.map(({ role, content, turn_id, sources, attachments }) => ({
        role,
        content,
        turnId: turn_id,
        attachments: attachments ?? [],
        metadata:
          role === "assistant"
            ? {
                answer: content,
                sources: sources ?? [],
                tool_calls: [],
                turn_id,
                attachments: attachments ?? [],
              }
            : undefined,
      }));
      const latestAssistantWithSources = [...loadedChat]
        .reverse()
        .find((item) => item.metadata?.sources.length);

      setConversationId(id);
      setChat(loadedChat);
      setToolCalls(savedToolCalls);
      setSelectedResponse(latestAssistantWithSources?.metadata ?? null);
      setSelectedTurnId(latestAssistantWithSources?.turnId ?? null);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleNewConversation() {
    setIsBusy(true);
    try {
      const conversation = await createConversation();
      setConversationId(conversation.conversation_id);
      setChat([]);
      setToolCalls([]);
      setSelectedResponse(null);
      setSelectedTurnId(null);
      clearSelectedImage();
      await refreshConversations();
    } finally {
      setIsBusy(false);
    }
  }

  async function handleDeleteConversation(conversation: ConversationInfo) {
    const confirmed = window.confirm(
      `“${conversation.title}” sohbeti kalıcı olarak silinsin mi?`,
    );
    if (!confirmed) return;

    setIsBusy(true);
    try {
      await deleteConversation(conversation.conversation_id);
      const remaining = await listConversations();
      setConversations(remaining);

      if (conversation.conversation_id !== conversationId) return;

      if (remaining.length > 0) {
        await openConversation(remaining[0].conversation_id);
      } else {
        await handleNewConversation();
      }
    } finally {
      setIsBusy(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = message.trim();
    if (!trimmed && !selectedImage) return;

    let activeConversationId = conversationId;
    if (!activeConversationId) {
      const conversation = await createConversation();
      activeConversationId = conversation.conversation_id;
      setConversationId(activeConversationId);
    }

    const outgoingImage = selectedImage;
    const outgoingPreview = selectedImagePreview;
    const userContent = trimmed || "Görsel eklendi.";

    setMessage("");
    setSelectedImage(null);
    setSelectedImagePreview(null);
    setChat((items) => [
      ...items,
      {
        role: "user",
        content: userContent,
        attachments: outgoingPreview
          ? [
              {
                filename: outgoingImage?.name ?? "görsel",
                content_type: outgoingImage?.type ?? "image/*",
                url: outgoingPreview,
              },
            ]
          : [],
      },
    ]);
    setIsBusy(true);

    try {
      const response = await sendMessage(userContent, activeConversationId, outgoingImage);
      setSelectedResponse(response);
      setSelectedTurnId(response.turn_id ?? null);
      setChat((items) => {
        const updatedItems = [...items];
        let lastUserIndex = -1;

        for (let index = updatedItems.length - 1; index >= 0; index -= 1) {
          if (updatedItems[index].role === "user") {
            lastUserIndex = index;
            break;
          }
        }

        if (lastUserIndex >= 0) {
          updatedItems[lastUserIndex] = {
            ...updatedItems[lastUserIndex],
            turnId: response.turn_id,
            attachments: response.attachments ?? updatedItems[lastUserIndex].attachments,
          };
        }

        return [
          ...updatedItems,
          {
            role: "assistant",
            content: response.answer,
            turnId: response.turn_id,
            metadata: response,
          },
        ];
      });
      const savedToolCalls = await getConversationToolCalls(activeConversationId);
      setToolCalls(savedToolCalls);
      await refreshConversations();
    } catch {
      setChat((items) => [
        ...items,
        { role: "assistant", content: "Backend ile bağlantı kurulamadı. Sunucunun çalıştığını kontrol et." },
      ]);
    } finally {
      setIsBusy(false);
    }
  }

  function handleMessageKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }

    if (isBusy || (!message.trim() && !selectedImage)) {
      return;
    }

    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }

  function handleImageChange(file?: File) {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      window.alert("Lütfen bir görsel dosyası seç.");
      return;
    }

    clearSelectedImage();
    setSelectedImage(file);
    setSelectedImagePreview(URL.createObjectURL(file));
    window.setTimeout(() => messageInputRef.current?.focus(), 0);
  }

  function clearSelectedImage() {
    setSelectedImage(null);
    setSelectedImagePreview((preview) => {
      if (preview) URL.revokeObjectURL(preview);
      return null;
    });
  }

  const reportUpdatedAt = evaluationStatus?.report.updated_at
    ? new Date(evaluationStatus.report.updated_at * 1000).toISOString()
    : null;
  const selectedRagasProvider =
    appSettings?.ragas_llm.providers.find((provider) => provider.id === ragasProvider) ??
    evaluationStatus?.ragas_llm.providers.find(
      (provider) => provider.id === ragasProvider,
    );
  const selectedModelOptions =
    appSettings?.ragas_llm.providers.find((provider) => provider.id === ragasProvider)?.models ??
    (ragasModel ? [ragasModel] : []);
  const selectedChatProvider = appSettings?.chat_llm.providers.find(
    (provider) => provider.id === chatProvider,
  );
  const selectedChatModelOptions = selectedChatProvider?.models ?? (chatModel ? [chatModel] : []);
  const ragasCommand = `backend/.venv/bin/python -u evaluation/run_ragas_eval.py \\
  --llm-provider ${ragasProvider} \\
  --eval-model ${ragasModel} \\
  --use-chat-history \\
  --ids q001,q002,q007,q008,q012 \\
  --report evaluation/reports/ragas_report_v1.json`;

  async function applyChatLlmSettings(nextProvider: RagasProvider, nextModel: string) {
    setSettingsMessage("Ayar uygulanıyor...");
    setChatProvider(nextProvider);
    setChatModel(nextModel);

    try {
      const settings = await updateChatLlmSettings(nextProvider, nextModel);
      setAppSettings(settings);
      setChatProvider(settings.chat_llm.active_provider);
      setChatModel(settings.chat_llm.active_model);
      setSettingsMessage("Chat modeli güncellendi.");
    } catch {
      setSettingsMessage("Ayar uygulanamadı. API key veya backend bağlantısını kontrol et.");
      try {
        const settings = await getAppSettings();
        setAppSettings(settings);
        setChatProvider(settings.chat_llm.active_provider);
        setChatModel(settings.chat_llm.active_model);
      } catch {
        // Keep the visible selection; the status message already explains the failure.
      }
    }
  }

  if (activePage === "home") {
    return (
      <main className="launch-shell">
        <section className="launch-panel" aria-label="Giriş">
          <div className="launch-copy">
            <p className="eyebrow">Manufacturing Maintenance Agent</p>
            <h1>Bakım asistanı</h1>
          </div>

          <div className="launch-actions" aria-label="Ekran seçimi">
            <button
              type="button"
              className="launch-card"
              onClick={() => setActivePage("chat")}
            >
              <MessageSquare size={28} />
              <span>
                <strong>Chat</strong>
                <small>Bakım dokümanlarıyla soru-cevap</small>
              </span>
            </button>

            <button
              type="button"
              className="launch-card"
              onClick={() => setActivePage("ragas")}
            >
              <BarChart3 size={28} />
              <span>
                <strong>Ragas</strong>
                <small>Golden dataset ve metrik ekranı</small>
              </span>
            </button>
          </div>
        </section>
      </main>
    );
  }

  if (activePage === "ragas") {
    return (
      <main className="ragas-page-shell">
        <section className="workspace">
          <header className="topbar">
            <div>
              <p className="eyebrow">Evaluation</p>
              <h1>Ragas değerlendirme</h1>
            </div>
            <div className="topbar-actions">
              <button
                type="button"
                className="secondary-button"
                onClick={() => setActivePage("home")}
              >
                <Home className="home-button-icon" size={16} />
                Giriş
              </button>
              <button
                type="button"
                className="upload-button"
                onClick={() => void getEvaluationStatus().then(setEvaluationStatus)}
                disabled={isEvaluationLoading}
              >
                <BarChart3 size={17} />
                Yenile
              </button>
            </div>
          </header>

          <section className="ragas-panel" aria-label="Ragas">
            {isEvaluationLoading && !evaluationStatus ? (
              <div className="empty-state">
                <strong>Ragas bilgileri yükleniyor.</strong>
              </div>
            ) : evaluationStatus ? (
              <>
                <div className="ragas-stat-grid">
                  <article className="ragas-stat">
                    <span>Golden dataset</span>
                    <strong>{evaluationStatus.total_questions}</strong>
                    <small>soru</small>
                  </article>
                  <article className="ragas-stat">
                    <span>Kategori</span>
                    <strong>{Object.keys(evaluationStatus.categories).length}</strong>
                    <small>başlık</small>
                  </article>
                  <article className="ragas-stat">
                    <span>Rapor</span>
                    <strong>{evaluationStatus.report.exists ? "Hazır" : "Yok"}</strong>
                    <small>{evaluationStatus.report.path}</small>
                  </article>
                </div>

                <section className="ragas-section">
                  <h2>Kategoriler</h2>
                  <div className="category-grid">
                    {Object.entries(evaluationStatus.categories).map(([name, count]) => (
                      <div className="category-pill" key={name}>
                        <span>{name}</span>
                        <strong>{count}</strong>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="ragas-section question-table">
                  <h2>Golden sorular</h2>
                  <div className="question-list">
                    {evaluationStatus.questions.map((question) => (
                      <article className="question-row" key={question.id}>
                        <strong>{question.id}</strong>
                        <span>{question.question}</span>
                        <small>{question.category} · {question.expected_behavior}</small>
                      </article>
                    ))}
                  </div>
                </section>
              </>
            ) : (
              <div className="empty-state">
                <strong>Ragas bilgileri alınamadı.</strong>
                <span>Backend bağlantısını kontrol et.</span>
              </div>
            )}
          </section>
        </section>

        <aside className="inspector">
          <section>
            <h2>Ragas raporu</h2>
            {evaluationStatus?.report.exists ? (
              <>
                <div className="source-row">
                  <strong>{evaluationStatus.report.path}</strong>
                  <span>
                    {evaluationStatus.report.size_bytes} byte
                    {reportUpdatedAt ? ` · ${formatDateTime(reportUpdatedAt)}` : ""}
                  </span>
                </div>
                {evaluationStatus.report.scores ? (
                  <div className="ragas-score-list">
                    {Object.entries(evaluationStatus.report.scores).map(([name, score]) => (
                      <div className="source-row" key={name}>
                        <strong>{name}</strong>
                        <span>{typeof score === "number" ? score.toFixed(3) : "Yok"}</span>
                      </div>
                    ))}
                  </div>
                ) : null}
              </>
            ) : (
              <p>Henüz rapor üretilmedi.</p>
            )}
          </section>

          <section>
            <h2>Çalıştırma</h2>
            <pre>{ragasCommand}</pre>
            {selectedRagasProvider ? (
              <p className="provider-note">
                {selectedRagasProvider.label}: {selectedRagasProvider.model}
                {selectedRagasProvider.configured ? "" : " · API key gerekli"}
              </p>
            ) : null}
          </section>

          <section>
            <h2>Davranışlar</h2>
            {evaluationStatus ? (
              Object.entries(evaluationStatus.expected_behaviors).map(([name, count]) => (
                <div className="source-row" key={name}>
                  <strong>{name}</strong>
                  <span>{count} soru</span>
                </div>
              ))
            ) : (
              <p>Veri bekleniyor.</p>
            )}
          </section>
        </aside>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="history-sidebar" aria-label="Sohbet geçmişi">
        <>
            <button
              type="button"
              className="secondary-button"
              onClick={() => setActivePage("home")}
            >
              <Home className="home-button-icon" size={15} />
              Giriş
            </button>

            <div className="chat-sidebar-tabs" aria-label="Chat menüsü">
              <button
                type="button"
                className={chatSidebarTab === "history" ? "chat-sidebar-tab active" : "chat-sidebar-tab"}
                onClick={() => setChatSidebarTab("history")}
              >
                <MessageSquare size={15} />
                Sohbetler
              </button>
              <button
                type="button"
                className={chatSidebarTab === "settings" ? "chat-sidebar-tab active" : "chat-sidebar-tab"}
                onClick={() => setChatSidebarTab("settings")}
              >
                <Settings size={15} />
                Ayarlar
              </button>
            </div>

            {chatSidebarTab === "history" ? (
              <>
            <button
              type="button"
              className="new-chat-button"
              onClick={() => void handleNewConversation()}
              disabled={isBusy}
            >
              <Plus size={17} />
              Yeni sohbet
            </button>

            <div className="conversation-list">
              {conversations.map((conversation) => (
                <div
                  className={
                    conversation.conversation_id === conversationId
                      ? "conversation-item active"
                      : "conversation-item"
                  }
                  key={conversation.conversation_id}
                >
                  <button
                    type="button"
                    className="conversation-open-button"
                    onClick={() => void openConversation(conversation.conversation_id)}
                    disabled={isBusy}
                  >
                    <MessageSquare className="conversation-icon" size={15} />
                    <span>
                      <strong>{conversation.title}</strong>
                      <small>
                        {formatDateTime(conversation.updated_at)}
                      </small>
                    </span>
                  </button>
                  <button
                    type="button"
                    className="conversation-delete-button"
                    aria-label={`${conversation.title} sohbetini sil`}
                    title="Sohbeti sil"
                    onClick={() => void handleDeleteConversation(conversation)}
                    disabled={isBusy}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
            </>
            ) : (
              <section className="chat-settings-panel" aria-label="Ayarlar">
                <div>
                  <h2>Chat ayarları</h2>
                  <span>Aktif çalışma değerleri</span>
                </div>
                {appSettings ? (
                  <div className="settings-list">
                    <section className="settings-group">
                      <h2>LLM provider</h2>
                      <div className="provider-toggle" role="group" aria-label="LLM provider">
                        {appSettings.chat_llm.providers.map((provider) => (
                          <button
                            type="button"
                            className={provider.id === chatProvider ? "provider-option active" : "provider-option"}
                            key={provider.id}
                            onClick={() => void applyChatLlmSettings(provider.id, provider.model)}
                          >
                            <strong>{provider.label}</strong>
                            <span>{provider.model}</span>
                            <small>{provider.configured ? "Hazır" : "API key gerekli"}</small>
                          </button>
                        ))}
                      </div>
                    </section>

                    <label className="settings-field">
                      <span>Model</span>
                      <select
                        value={chatModel}
                        onChange={(event) => void applyChatLlmSettings(chatProvider, event.target.value)}
                      >
                        {Array.from(new Set(selectedChatModelOptions)).map((model) => (
                          <option key={model} value={model}>
                            {model}
                          </option>
                        ))}
                      </select>
                    </label>

                    {settingsMessage ? (
                      <p className="provider-note">{settingsMessage}</p>
                    ) : null}

                    <div className="settings-row">
                      <span>Chat modeli</span>
                      <strong>{appSettings.chat_model}</strong>
                    </div>
                    <div className="settings-row">
                      <span>Chat provider</span>
                      <strong>{appSettings.chat_llm.active_provider}</strong>
                    </div>
                    <div className="settings-row">
                      <span>Vision modeli</span>
                      <strong>{appSettings.vision_model}</strong>
                    </div>
                    <div className="settings-row">
                      <span>Embedding</span>
                      <strong>{appSettings.embedding_model}</strong>
                    </div>
                    <div className="settings-row">
                      <span>Top K</span>
                      <strong>{appSettings.top_k}</strong>
                    </div>
                    <div className="settings-row">
                      <span>Rerank</span>
                      <strong>{appSettings.rerank_top_n}</strong>
                    </div>
                    <div className="settings-row">
                      <span>Chunk</span>
                      <strong>{appSettings.chunk_size} / {appSettings.chunk_overlap}</strong>
                    </div>
                  </div>
                ) : (
                  <p>Ayarlar yükleniyor.</p>
                )}
              </section>
            )}
          </>
      </aside>

      <section className="workspace">
        {activePage === "chat" ? (
          <>
            <header className="topbar">
              <div>
                <p className="eyebrow">Manufacturing Maintenance Agent</p>
                <h1>Bakım asistanı</h1>
              </div>
              <FileUpload />
            </header>

            <section className="chat-panel" aria-label="Sohbet">
              <section className="chat-window" ref={chatWindowRef}>
                {chat.length === 0 ? (
                  <div className="empty-state">
                    <strong>Alarm, bakım, güvenlik veya tarih sorusu sor.</strong>
                    <span>Örnek: E42 alarm kodu ne anlama geliyor?</span>
                  </div>
                ) : (
                  chat.map((item, index) => {
                    const itemToolCalls = item.turnId
                      ? toolCalls.filter((tool) => tool.turn_id === item.turnId)
                      : [];
                    const isSelectableAssistant = item.role === "assistant" && item.turnId;
                    const isSelected = item.turnId && item.turnId === selectedTurnId;

                    return (
                      <article
                        className={[
                          "message",
                          item.role,
                          isSelectableAssistant ? "selectable" : "",
                          isSelected ? "selected" : "",
                        ].filter(Boolean).join(" ")}
                        key={`${item.role}-${index}`}
                        onClick={() => {
                          if (!isSelectableAssistant) return;
                          setSelectedTurnId(item.turnId ?? null);
                          setSelectedResponse(item.metadata ?? null);
                        }}
                      >
                        <span className="message-author">
                          {item.role === "user" ? "Sen" : "Asistan"}
                        </span>
                        <MarkdownMessage content={item.content} />
                        {item.attachments?.length ? (
                          <div className="message-attachments">
                            {item.attachments.map((attachment) => (
                              <figure className="message-image" key={attachment.url}>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    setViewedImage(attachment);
                                  }}
                                  aria-label={`${attachment.filename} görselini büyüt`}
                                >
                                  <img
                                    src={resolveAssetUrl(attachment.url)}
                                    alt={attachment.filename}
                                  />
                                </button>
                                <figcaption>{attachment.filename}</figcaption>
                              </figure>
                            ))}
                          </div>
                        ) : null}
                        {item.role === "assistant" && itemToolCalls.length > 0 ? (
                          <details className="message-tool-calls">
                            <summary>{itemToolCalls.length} tool çağrısı</summary>
                            {itemToolCalls.map((tool) => (
                              <pre key={tool.id}>{JSON.stringify(tool, null, 2)}</pre>
                            ))}
                          </details>
                        ) : null}
                      </article>
                    );
                  })
                )}
              </section>

              <form className="composer" onSubmit={handleSubmit}>
                <div className="composer-main">
                  {selectedImagePreview ? (
                    <div className="image-preview-pill">
                      <img src={selectedImagePreview} alt={selectedImage?.name ?? "Seçili görsel"} />
                      <span>{selectedImage?.name}</span>
                      <button
                        type="button"
                        onClick={clearSelectedImage}
                        aria-label="Görseli kaldır"
                        disabled={isBusy}
                      >
                        <X size={13} />
                      </button>
                    </div>
                  ) : null}
                  <input
                    ref={messageInputRef}
                    value={message}
                    onChange={(event) => setMessage(event.target.value)}
                    onKeyDown={handleMessageKeyDown}
                    placeholder="Sorunu yaz..."
                    disabled={isBusy}
                  />
                </div>
                <label className="image-attach-button" aria-label="Görsel ekle">
                  <ImagePlus size={18} />
                  <input
                    type="file"
                    accept="image/*"
                    disabled={isBusy}
                    onChange={(event) => {
                      handleImageChange(event.target.files?.[0]);
                      event.target.value = "";
                    }}
                  />
                </label>
                <button type="submit" disabled={isBusy} aria-label="Mesaj gönder">
                  <Send className={isBusy ? "send-icon spinning" : "send-icon"} size={18} />
                </button>
              </form>
            </section>
          </>
        ) : (
          <>
            <header className="topbar">
              <div>
                <p className="eyebrow">Evaluation</p>
                <h1>Ragas değerlendirme</h1>
              </div>
              <button
                type="button"
                className="upload-button"
                onClick={() => void getEvaluationStatus().then(setEvaluationStatus)}
                disabled={isEvaluationLoading}
              >
                <BarChart3 size={17} />
                Yenile
              </button>
            </header>

            <section className="ragas-panel" aria-label="Ragas">
              {isEvaluationLoading && !evaluationStatus ? (
                <div className="empty-state">
                  <strong>Ragas bilgileri yükleniyor.</strong>
                </div>
              ) : evaluationStatus ? (
                <>
                  <div className="ragas-stat-grid">
                    <article className="ragas-stat">
                      <span>Golden dataset</span>
                      <strong>{evaluationStatus.total_questions}</strong>
                      <small>soru</small>
                    </article>
                    <article className="ragas-stat">
                      <span>Kategori</span>
                      <strong>{Object.keys(evaluationStatus.categories).length}</strong>
                      <small>başlık</small>
                    </article>
                    <article className="ragas-stat">
                      <span>Rapor</span>
                      <strong>{evaluationStatus.report.exists ? "Hazır" : "Yok"}</strong>
                      <small>{evaluationStatus.report.path}</small>
                    </article>
                  </div>

                  <section className="ragas-section">
                    <h2>Kategoriler</h2>
                    <div className="category-grid">
                      {Object.entries(evaluationStatus.categories).map(([name, count]) => (
                        <div className="category-pill" key={name}>
                          <span>{name}</span>
                          <strong>{count}</strong>
                        </div>
                      ))}
                    </div>
                  </section>

                  <section className="ragas-section question-table">
                    <h2>Golden sorular</h2>
                    <div className="question-list">
                      {evaluationStatus.questions.map((question) => (
                        <article className="question-row" key={question.id}>
                          <strong>{question.id}</strong>
                          <span>{question.question}</span>
                          <small>{question.category} · {question.expected_behavior}</small>
                        </article>
                      ))}
                    </div>
                  </section>
                </>
              ) : (
                <div className="empty-state">
                  <strong>Ragas bilgileri alınamadı.</strong>
                  <span>Backend bağlantısını kontrol et.</span>
                </div>
              )}
            </section>
          </>
        )}
      </section>

      <aside className="inspector">
        {activePage === "chat" ? (
          <>
          <section>
          <h2>Kaynaklar</h2>
          {displayedSources.length ? (
            displayedSources.map((source) => (
              <div className="source-row" key={source.chunk_id}>
                <strong>{source.document_name}</strong>
                <span>
                  {source.chunk_id}
                  {source.page_number ? ` · Sayfa ${source.page_number}` : ""}
                  {" · "}
                  {source.score.toFixed(2)}
                </span>
              </div>
            ))
          ) : (
            <p>Kaynak bulunmadı.</p>
          )}
        </section>

        <details className="tool-call-section">
          <summary className="tool-call-header">
            <div>
              <h2>Tool çağrıları</h2>
              <span>{selectedTurnId ? "Seçili cevap" : "Tüm sohbet"}</span>
            </div>
          </summary>

          {selectedTurnId ? (
            <button
              type="button"
              className="show-all-tools-button"
              onClick={() => {
                setSelectedTurnId(null);
                setSelectedResponse(null);
              }}
            >
              Tümünü göster
            </button>
          ) : null}

          {selectedTurnId ? (
            selectedToolCalls.length ? (
              <div className="tool-call-list">
                {selectedToolCalls.map((tool) => (
                  <article className="tool-call-item" key={tool.id}>
                    <time dateTime={tool.created_at}>
                      {formatDateTime(tool.created_at)}
                    </time>
                    <pre>{JSON.stringify(tool, null, 2)}</pre>
                  </article>
                ))}
              </div>
            ) : (
              <p>Bu cevap için tool çağrısı yok.</p>
            )
          ) : turnToolSummaries.length ? (
            <div className="tool-call-list">
              {turnToolSummaries.map((turn) => (
                <article className="turn-tool-group" key={turn.turnId}>
                  <div className="turn-tool-question">
                    <strong>{turn.userMessage}</strong>
                    <span>
                      {turn.toolCalls.length
                        ? `${turn.toolCalls.length} tool çağrısı`
                        : "Tool çağrısı yok"}
                    </span>
                  </div>
                  {turn.toolCalls.length ? (
                    turn.toolCalls.map((tool) => (
                      <div className="tool-call-item" key={tool.id}>
                        <time dateTime={tool.created_at}>
                          {formatDateTime(tool.created_at)}
                        </time>
                        <pre>{JSON.stringify(tool, null, 2)}</pre>
                      </div>
                    ))
                  ) : (
                    <p className="no-tool-call">Bu mesajda agent tool kullanmadı.</p>
                  )}
                </article>
              ))}
            </div>
          ) : (
            <p>Henüz cevap yok.</p>
          )}
        </details>
        </>
        ) : (
          <>
            <section>
              <h2>Ragas raporu</h2>
              {evaluationStatus?.report.exists ? (
                <>
                  <div className="source-row">
                    <strong>{evaluationStatus.report.path}</strong>
                    <span>
                      {evaluationStatus.report.size_bytes} byte
                      {reportUpdatedAt ? ` · ${formatDateTime(reportUpdatedAt)}` : ""}
                    </span>
                  </div>
                  {evaluationStatus.report.scores ? (
                    <div className="ragas-score-list">
                      {Object.entries(evaluationStatus.report.scores).map(([name, score]) => (
                        <div className="source-row" key={name}>
                          <strong>{name}</strong>
                          <span>{typeof score === "number" ? score.toFixed(3) : "Yok"}</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </>
              ) : (
                <p>Henüz rapor üretilmedi.</p>
              )}
            </section>

            <section>
              <h2>Çalıştırma</h2>
              <pre>{ragasCommand}</pre>
            </section>

            <section>
              <h2>Davranışlar</h2>
              {evaluationStatus ? (
                Object.entries(evaluationStatus.expected_behaviors).map(([name, count]) => (
                  <div className="source-row" key={name}>
                    <strong>{name}</strong>
                    <span>{count} soru</span>
                  </div>
                ))
              ) : (
                <p>Veri bekleniyor.</p>
              )}
            </section>
          </>
        )}
      </aside>
      {viewedImage ? (
        <div
          className="image-viewer-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label={viewedImage.filename}
          onClick={() => setViewedImage(null)}
        >
          <figure
            className="image-viewer"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              className="image-viewer-close"
              onClick={() => setViewedImage(null)}
              aria-label="Görseli kapat"
            >
              <X size={18} />
            </button>
            <img src={resolveAssetUrl(viewedImage.url)} alt={viewedImage.filename} />
            <figcaption>{viewedImage.filename}</figcaption>
          </figure>
        </div>
      ) : null}
    </main>
  );
}
