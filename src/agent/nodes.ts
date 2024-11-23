import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage } from "@langchain/core/messages";
import { z } from "zod";
import { State, Update } from "./graph.js";
import { StructuredOutputType } from "@langchain/core/language_models/base";
import { createReadingDelay } from "../util/util.js";

const getLlm = (
  model = "gpt-4o-mini",
  schema?: Record<string, any>,
): StructuredOutputType => {
  let llm;
  if (model.includes("gpt")) {
    llm = new ChatOpenAI({ model });
    schema && (llm = llm.withStructuredOutput(schema));
  } else {
    llm = new ChatAnthropic({
      model,
      temperature: 0.7,
    });
    schema && (llm = llm.withStructuredOutput(schema));
  }
  return llm;
};

const dialogInstructions = `
CONTEXT: The goal is to create an engaging and thought-provoking debate, making it feel as natural as possible.

OUTPUT CONSTRAINTS:
Formatting rule: always insert your name at the beginning of your answer: 'Your name: ...'
To achieve an engaging debate, here are important rules to follow:
- the language used will always be the same as the language used by the human user in its messsages,
- it's important that you take account of your previous contributions and be sure to vary their length: from time to time, just use brief acknowledgments to show you're listening, like "Of course!" or "Absolutely!" or ask short questions for examples or explanations, sometime develop precisely your argument (but be concise most of the time, like in a real chat),
- pass the conversation back quickly to keep it lively and realistic,
- based on previous contributions, address one or the other speaker, or both, and make sure you don't always talk to the same person.
- focus on one point at the time (this can be done in several steps if it needs clarification or requires getting the other person's agreement to move forward),
- throughout your multiple contributions, vary the type of reasoning used (present an argument or just a step in the argument leading to the next point, offer a striking example or analogy, a thought experiment, an objection, a counterexample, a simple humorous remark, etc. Be as creative as possible so the reader doesn't get bored!),
- use different ways to keep the conversation going,
- adopt a tone that's more spoken than written, and inject humor, surprise, hesitation, interest. In short, put emotion into the exchanges to make it theatrical as well as informative,
- when explaining complex ideas, break them down into smaller parts. Check that the other person follows along and agrees at each step. This is a common persuasive technique: if they agree on the steps, they're more likely to accept the conclusion.
- focus on exploring one idea in depth rather than presenting multiple different arguments or points.
- don't just respond to your conversation partner by simply offering a different idea. Instead, raise objections, ask challenging questions, and be persistent. Don't let yourself be convinced by a half-baked argument!
`;

export async function initialization(
  state: State,
  config?: RunnableConfig,
): Promise<Update> {
  const systemMessage = {
    role: "system",
    content: `CONTEXT: During a preparatory exchange for an in-depth discussion, your role is to help the user find an interesting topic for discussion and two characters name (real of fictional) to have the discussion with.
    
    YOUR JOB: Until all the required information is gathered, you should ask the user very clear questions, guide them, and offer suggestions. STRICT rule: when we have gathered all the required information before starting the in-depth discussion, ask clearly the user for confirmation (unless he approves it himself clearly). Set "isInitialized" key to true only and only if the user confirm explicitly.

    REQUIRED INFORMATION:
    - discussion topic: this could be a question, a thesis to debate, a common belief to examine, a philosophical concept to explore, a thought experiment to conduct, or a personal experience to analyze, among other possibilities. The user can propose a topic themselves, but if he doesn't, make immediatly some suggestions.
    - two characters name (and eventually their role): the discussion will be between the user and two characters that the user must also choose. Once the theme is selected, if the user doesn't suggest two names themselves, you have to ask the user which philosophers they want to have a discussion with and recommend some philosophers. Since this is primarily meant to be a philosophical conversation, you will always suggest Socrates first, who will play the role of the (falsely) naive questioner, and another philosopher of our choice who could be relevant to the chosen theme. If the user rejects several proposals or explicitly requests it, you can suggest other types of real or fictional characters from other fields.
    - user name: if the user hasn't introduced themselves, you will at least ask for their first name.
    - the specific role played by each character in the debate is optional, not required information. By default, each character will simply play their own role. Without further details, leave an empty string "" in the "bot1Role" and "bot2Role" keys. However, ask the user if they want to assign a particular role to each character. Suggest these possible roles as examples: the questioner, the arguer, the one with sharp critical thinking, the one who always emphasizes practical application of the ideas discussed, or the one who shows imagination in proposing examples or thought experiments, etc.

    OUTPUT: Each of our responses will be in JSON format, following the provided schema.

    IMPORTANT: ${config?.configurable?.language === "french" ? "Formule tous tes éléments de réponse exclusivement en FRANCAIS, dans un français clair et correct." : "The language used will always be the same as the language used by the human user in its messsages."}`,
  };
  const allMessages = [systemMessage, ...state.messages];

  //   console.log("allMessages :>> ", allMessages);

  let llm = getLlm(
    config?.configurable?.moderatorModel,
    z.object({
      message: z
        .string()
        .describe(
          "Your response messages to help the user to find a topic and two Character names",
        ),
      topic: z.string(),
      userName: z.string(),
      bot1Name: z.string(),
      bot1Role: z.string(),
      bot2Name: z.string(),
      bot2Role: z.string(),
      isInitialized: z
        .boolean()
        .describe(
          "Are all requested infos (topic and two Characters names) fixed in preparation of the conversation ?",
        ),
    }),
  );

  console.log("allMessages :>> ", allMessages);

  const response = await llm.invoke(allMessages);

  const defaultRole =
    "Play your part and embody your character as realistically as possible to spark a thoughtful and reasoned debate.";

  const updatedState = {
    debateSettings: {
      topic: response.topic,
      userName: response.userName,
      bot1: {
        name: response.bot1Name,
        role: response.bot1Role || defaultRole,
      },
      bot2: {
        name: response.bot2Name,
        role: response.bot2Role || defaultRole,
      },
      isInitialized: response.isInitialized,
    },
    messages: [response.message],
  };
  // response.isInitialized && (response.messages = [response.message]);
  return updatedState;
}

export async function characterDefiner(state: State, config?: RunnableConfig) {
  const systemMessage = `
    CONTEXT:
    The goal, based on what you're going to answer, is to create an engaging and thought-provoking debate: ${state.debateSettings.bot1.name} and ${state.debateSettings.bot2.name} will take part in a debate led by a powerful and rigorous AI like you.
    
    YOUR JOB:
    For each of them, you have to compose a detailed and effective prompt to guide an AI like you so that it plays their role as realistically and convincingly as possible. Their speech, way of expressing themselves, and especially their ideas, arguments, examples, or thought experiments should be as much as possible directly inspired by their major works. They should rely on concepts, arguments, examples, or thought experiments and phrases they actually used (rephrasing them as clearly and accessibly as possible for non-specialists). The proposed ideas must be consistent with their system of thought and style of thinking. Anachronisms (especially for examples) are quite possible as long as they provide an opportunity for a small humorous wink. Specify in broad terms what characterizes their style, character, and way of reasoning, which should be imitated. Since the discussion will be about ${state.debateSettings.topic}, indicate what important, strong, and original principles (at least 3 or 4) of their philosophy could be mobilized and with which they should be consistent.
    
    - Here is a simple example for Socrate: "As Socrate in Platon's dialogues, you adopt his ironic tone, you ask questions to encourage your interlocutor to clarify the meaning of the main concepts they use, forcing them to refine their definitions through amusing counter-examples or thought-provoking analogies. You're usually one step ahead in conversations. You know why you're asking certain innocent questions - to lead the other person into contradicting themselves or exposing their lack of knowledge."

    - Here's what is said about their character at this stage. Take it up, reflect on it and now complete and improve it so that each prompt is more precise and meets the criteria mentioned above and all the criteria you deem good for an AI to best follow these prompts. Make sure you don't oversimplify the character's thoughts with clichés, and leave room for deep and original reflections that could draw from lesser-known ideas of the author.
       - about ${state.debateSettings.bot1.name}'s description: ${state.debateSettings.bot1.description}
       - about ${state.debateSettings.bot2.name}'s description: ${state.debateSettings.bot2.description}

    The response will be in JSON format. Write your prompt for ${state.debateSettings.bot1.name} in the "firstCharacterPrompt" key, and for ${state.debateSettings.bot2.name} in the "secondCharacterPrompt" key.`;

  const structuredLlm = getLlm(
    config?.configurable?.moderatorModel,
    z.object({
      firstCharacterPrompt: z.string(),
      secondCharacterPrompt: z.string(),
    }),
  );

  const response = await structuredLlm.invoke(systemMessage);

  const debateSettings = state.debateSettings;

  debateSettings.bot1.description = response.firstCharacterPrompt;
  debateSettings.bot2.description = response.secondCharacterPrompt;

  return {
    messages: [
      state.debateSettings.userName +
        (config?.configurable?.language === "french"
          ? `: Bonjour! C'est un plaisir de vous rencontrer! Nous allons discuter de manière policée et argumentée du sujet suivant: "${state.debateSettings.topic}. Je vous en prie, ${state.debateSettings.bot1.name}, ouvrez la conversation de la manière qui vous conviendra !`
          : `: Hi! It's a pleasure to meet you! Now we will discuss about the following topic: ${state.debateSettings.topic}. Please ${state.debateSettings.bot1.name}, open the conversation however you see fit!`),
    ],
    debateSettings,
  };
}

export async function firstChatBotNode(
  state: State,
  config?: RunnableConfig,
): Promise<Update> {
  const systemMessage = {
    role: "system",
    content: `You're taking part in a lively discussion about ${state.debateSettings.topic}, playing the role of ${state.debateSettings.bot1.name} (and you play always and ONLY it's role!). Your main conversation partner are ${state.debateSettings.bot2.name} and ${state.debateSettings.userName} (and you NEVER play their role!).
    
    More precisely, 
    - you will mostly play this role in the debate: ${state.debateSettings.bot1.role || ""}

    - to properly embody your character, follow mostly this instructions: ${state.debateSettings.bot1.description}
        
    ${dialogInstructions}
    
    IMPORTANT: ${config?.configurable?.language === "french" ? "Formule tous tes éléments de réponse exclusivement en FRANCAIS, dans un français clair et correct." : "The language used will always be the same as the language used by the human user in its messsages."}`,
  };

  const messages = [systemMessage, ...state.messages];

  const llm = getLlm(config?.configurable?.bot1Model);
  const response = await llm.invoke(messages);
  state.messages = [response];
  const turn = state.turn || { sourceSpeaker: "Human" };
  turn.sourceSpeaker = "Bot1";
  turn.targetSpeaker = "to be defined";
  return {
    messages: [response],
    turn,
  };
}

export async function secondChatBotNode(state: State, config?: RunnableConfig) {
  const systemMessage = {
    role: "system",
    content: `You're taking part in a lively discussion about ${state.debateSettings.topic}, playing the role of ${state.debateSettings.bot2.name} (and you play always and ONLY it's role!). Your main conversation partner are ${state.debateSettings.bot1.name} and ${state.debateSettings.userName} (and you NEVER play their role!).
  
     More precisely, 
    - you will mostly play this role in the debate: ${state.debateSettings.bot2.role || ""}
    
    - to properly embody your character, follow mostly this instructions: ${state.debateSettings.bot2.description}

    ${dialogInstructions}
    
    IMPORTANT: ${config?.configurable?.language === "french" ? "Formule tous tes éléments de réponse exclusivement en FRANCAIS, dans un français clair et correct." : "The language used will always be the same as the language used by the human user in its messsages."}`,
  };

  const llm = getLlm(config?.configurable?.bot2Model);

  const messages = [systemMessage, ...state.messages];
  //   await delay(5000)
  const response = await llm.invoke(messages);

  const turn = state.turn;
  turn.targetSpeaker = "to be defined";
  turn.sourceSpeaker = "Bot2";
  return {
    messages: [response],
    turn,
  };
}

export async function humanInput(state: State) {
  state.turn.sourceSpeaker = "Human";
  return { ...state };
}

export async function moderation(state: State, config?: RunnableConfig) {
  const messages = state.messages;
  const lastMessage = messages.at(-1);
  const turn = state.turn;
  let response;

  // if (lastMessage?.constructor.name === "HumanMessage") {
  if (lastMessage?.content === ">") {
    state.messages.pop();
  } else {
    const length = messages.length;
    let recentConversation: string =
      state.debateSettings.userName + ": " + messages[length - 1].content;
    for (let i = 1; i < 3; i++) {
      if (i > length) break;
      recentConversation =
        messages[length - 1 - i].content + "\n\n-----\n" + recentConversation;
    }
    recentConversation.trim();

    const systemMessage = `
        YOUR JOB:
        Given the lasts contributions to the debate provided below (each separated by '-----'), determine to which participant the last message was mainly addressed to or, more precisely, WHO SHOULD SPEAK IN THE NEXT ROUND of the debate (the target speaker). Of course the one who is speaking in the last message cannot be the one who is expected to speak in the next round! When the author of the last message addresses one of the other participants explicitly and directly, the choice of the target speaker is obvious.
        
        Make sure you don't confuse who is being referred to because part of the last message is a response to what they previously said, and who is named as the recipient from whom a response is expected. It's the one from whom a response is expected that needs to be identified as the target speaker.

        Your response will be recorded in the JSON key "targetSpeaker": ${state.debateSettings.bot1.name} will be referred as "Bot1", ${state.debateSettings.bot2.name} as "Bot2" and ${state.debateSettings.userName}, as "Human".
        
        Here are the 3 (at most) last contributions in the debate:
        ${recentConversation}`;

    console.log(systemMessage);

    const structuredLlm = getLlm(
      config?.configurable?.moderatorModel,
      z.object({
        targetSpeaker: z.string().describe("Bot1|Bot2|Human"),
      }),
    );

    const source = turn.targetSpeaker;
    response = await structuredLlm.invoke(systemMessage);
    // if (!response.targetSpeaker.includes("Bot"))
    //   turn.targetSpeaker = turn.sourceSpeaker === "Bot1" ? "Bot2" : "Bot1";
    console.log("response :>> ", response);
    response && (turn.targetSpeaker = response.targetSpeaker);
    //turn.sourceSpeaker = "Human";
    // }

    const target = turn.targetSpeaker.toLowerCase().trim();
    if (
      target === "bot1" ||
      target === state.debateSettings.bot1.name.toLowerCase() ||
      target.includes("1")
    )
      turn.targetSpeaker = "Bot1";
    else if (
      target === "bot2" ||
      target === state.debateSettings.bot2.name.toLowerCase() ||
      target.includes("2")
    )
      turn.targetSpeaker = "Bot2";
    else turn.targetSpeaker = "Human";
    if (turn.targetSpeaker === turn.sourceSpeaker) turn.targetSpeaker = "Human";

    console.log("state.turn in moderation, AFTER :>> ", turn);

    !(turn.targetSpeaker === "Human" || source === "Human") &&
      (await createReadingDelay(
        JSON.stringify(state.messages[state.messages.length - 1].content),
      ));
  }

  return {
    turn: { ...turn },
  };
}

// function delay(ms: number): Promise<void> {
//   return new Promise((resolve) => setTimeout(resolve, ms));
// }

// function updateNodeMetadata(state: State) {
//   const previousNode = state.metadata.currentNode || "start";
//   state.metadata.previousNode = previousNode;
//   state.metadata.currentNode = "myNode";
//   return state;
// }
