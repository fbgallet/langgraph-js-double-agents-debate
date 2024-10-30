import {
  StateGraph,
  START,
  END,
  Annotation,
  MemorySaver,
} from "@langchain/langgraph";
import { MessagesAnnotation } from "@langchain/langgraph";
import {
  firstChatBotNode,
  humanInput,
  initialization,
  moderation,
  roleDefiner,
  secondChatBotNode,
} from "./nodes.js";
import {
  initOrContinue,
  lauchConversation,
  nextSpeaker,
  shouldContinue,
  verifyInitialization,
} from "./edges.js";
import { BaseMessage } from "@langchain/core/messages";

export type Initialization = {
  // messages: BaseMessage[];
  conversationTopic: string;
  userName: string;
  firstCharacterName: string;
  firstCharacterRole: string;
  secondCharacterName: string;
  secondCharacterRole: string;
  isInitialized: boolean;
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
  initialization: Annotation<Initialization>,
  metadata: Annotation<Metadata>,
});

export type State = typeof StateAnnotation.State;
export type Update = typeof StateAnnotation.Update;

const config = { streamMode: "values" as const };

export const workflow = new StateGraph(StateAnnotation)
  // node pour définir le thème et les rôle,
  // qui boucle tant que le thème n'a pas été bien défini
  .addNode("Initialization", initialization)
  .addNode("RoleDefiner", roleDefiner)
  .addNode("Bot1", firstChatBotNode)
  .addNode("Bot2", secondChatBotNode)
  .addNode("Human", humanInput)
  .addNode("Moderator", moderation)

  // .addConditionalEdges(START, initOrContinue)
  .addEdge("__start__", "Initialization")
  .addConditionalEdges("Initialization", verifyInitialization)
  .addConditionalEdges("RoleDefiner", lauchConversation)
  .addEdge("Bot1", "Human")
  .addEdge("Bot2", "Human")
  .addEdge("Human", "Moderator")
  .addConditionalEdges("Moderator", nextSpeaker);

const graphStateMemory = new MemorySaver();

export const simulation = workflow.compile({
  checkpointer: graphStateMemory,
  interruptBefore: ["Human"],
});

simulation.name = "Auto-conversation Agent";
