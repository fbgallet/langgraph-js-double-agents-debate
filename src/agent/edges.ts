import { MessagesAnnotation } from "@langchain/langgraph";
import { State, StateAnnotation } from "./graph.js";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

let round = 0;

export function initOrContinue(state: State) {
  //   const messages = state.messages;
  //   console.log("State from initOrContinue", state);
  if (state.initialization?.isInitialized) return "Bot1";
  else {
    return "Initialization";
  }
}

export function nextSpeaker(state: State) {
  console.log(
    "state.initialization.sourceSpeaker :>> ",
    state.initialization.sourceSpeaker,
  );
  console.log(
    "state.initialization.targetSpeaker :>> ",
    state.initialization.targetSpeaker,
  );
  if (state.initialization.sourceSpeaker === "Human")
    return state.initialization.targetSpeaker;
  else {
    if (state.initialization.sourceSpeaker === "Bot1") return "Bot2";
    if (state.initialization.sourceSpeaker === "Bot2") return "Bot1";
    return "__end__";
  }
}

export function verifyInitialization(state: typeof StateAnnotation.State) {
  //   const messages = state.messages;
  console.log("state :>> ", state);
  if (state.initialization?.isInitialized) {
    return "RoleDefiner";
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
