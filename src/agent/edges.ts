import { MessagesAnnotation } from "@langchain/langgraph";
import { State, StateAnnotation } from "./graph.js";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { createReadingDelay } from "../util/util.js";

let round = 0;

export function initOrContinue(state: State) {
  //   const messages = state.messages;
  //   console.log("State from initOrContinue", state);
  if (state.debateSettings?.isInitialized) return "Bot1";
  else {
    return "DebateSetting";
  }
}

export async function nextSpeaker(state: State) {
  const target = state.turn.targetSpeaker;
  if (target === "Bot1" || target === "Bot2") return target;
  else return "Human";
  // let nextNode = "Human";
  // console.log("state.turn in nextSpeaker :>> ", state.turn);
  // const target = state.turn.targetSpeaker.toLowerCase().trim();
  // if (
  //   target === "bot1" ||
  //   target === state.debateSettings.bot1.name.toLowerCase() ||
  //   target.includes("1")
  // )
  //   nextNode = "Bot1";
  // else if (
  //   target === "bot2" ||
  //   target === state.debateSettings.bot2.name.toLowerCase() ||
  //   target.includes("2")
  // )
  //   nextNode = "Bot2";
  // else return nextNode;
  // return nextNode;
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
