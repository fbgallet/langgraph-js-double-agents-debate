import { MessagesAnnotation } from "@langchain/langgraph";
import { State, StateAnnotation } from "./graph.js";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

let round = 0;

export function initOrContinue(state: State) {
  //   const messages = state.messages;
  //   console.log("State from initOrContinue", state);
  if (state.debateSettings?.isInitialized) return "Bot1";
  else {
    return "DebateSetting";
  }
}

export function nextSpeaker(state: State) {
  if (state.turn.sourceSpeaker === "Human") return state.turn.targetSpeaker;
  else {
    if (state.turn.sourceSpeaker === "Bot1") return "Bot2";
    if (state.turn.sourceSpeaker === "Bot2") return "Bot1";
    return "__end__";
  }
}

export function verifyInitialization(state: typeof StateAnnotation.State) {
  //   const messages = state.messages;
  console.log("state :>> ", state);
  if (state.debateSettings?.isInitialized) {
    return "CharacterDefiner";
  } else {
    return "__end__";
  }
}

let reflexionOnRoles = true;
export function lauchConversation(state: typeof StateAnnotation.State) {
  console.log("state :>> ", state);
  round = 0;
  if (reflexionOnRoles) {
    // reflexionOnRoles = false;
    return "Bot1";
  } else {
    reflexionOnRoles = true;
    return "RoleDefiner";
  }
}
