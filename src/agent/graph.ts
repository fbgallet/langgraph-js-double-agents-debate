import {
  StateGraph,
  START,
  END,
  Annotation,
  MemorySaver,
} from "@langchain/langgraph";
import { MessagesAnnotation } from "@langchain/langgraph";
import {
  characterDefiner,
  firstChatBotNode,
  humanInput,
  initialization,
  moderation,
  secondChatBotNode,
} from "./nodes.js";
import {
  lauchConversation,
  nextSpeaker,
  verifyInitialization,
} from "./edges.js";

export type DebateSettings = {
  topic: string;
  userName: string;
  bot1: {
    name: string;
    description?: string;
    role?: string;
  };
  bot2: {
    name: string;
    description?: string;
    role?: string;
  };
  isInitialized: boolean;
};
export type Turn = {
  sourceSpeaker: string;
  targetSpeaker: string;
};

type Metadata = {
  currentNode: string;
  previousNode: string | null;
  nextNode: string | null;
};

export const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  debateSettings: Annotation<DebateSettings>,
  turn: Annotation<Turn>,
});

export type State = typeof StateAnnotation.State;
export type Update = typeof StateAnnotation.Update;

type AvailableModels =
  | "gpt-4o-mini"
  | "gpt-4o"
  | "claude-3-5-sonnet-20241022"
  | "claude-3-5-haiku-20241022";
export const defaultModel = "gpt-4o-mini";

type AvailableLanguages = "english" | "french";

const ConfigurableAnnotation = Annotation.Root({
  language: Annotation<AvailableLanguages>,
  bot1Model: Annotation<AvailableModels>,
  bot2Model: Annotation<AvailableModels>,
  moderatorModel: Annotation<AvailableModels>,
  botsOnly: Annotation<boolean>,
});

export const workflow = new StateGraph(StateAnnotation, ConfigurableAnnotation)
  // node pour définir le thème et les rôle,
  // qui boucle tant que le thème n'a pas été bien défini
  .addNode("DebateSetting", initialization)
  .addNode("CharacterDefiner", characterDefiner)
  .addNode("Bot1", firstChatBotNode)
  .addNode("Bot2", secondChatBotNode)
  .addNode("Human", humanInput)
  .addNode("Moderator", moderation)

  // .addConditionalEdges(START, initOrContinue)
  .addEdge(START, "DebateSetting")
  .addConditionalEdges("DebateSetting", verifyInitialization)
  .addConditionalEdges("CharacterDefiner", lauchConversation)
  .addEdge("Bot1", "Moderator")
  .addEdge("Bot2", "Moderator")
  .addEdge("Human", "Moderator")
  .addConditionalEdges("Moderator", nextSpeaker);

const graphStateMemory = new MemorySaver();

export const simulation = workflow.compile({
  checkpointer: graphStateMemory,
  interruptBefore: ["Human"],
});

simulation.name = "Auto-conversation Agent";
