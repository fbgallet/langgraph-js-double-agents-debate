import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage } from "@langchain/core/messages";
import { z } from "zod";
import { Initialization, State, Update } from "./graph.js";
import { StructuredOutputType } from "@langchain/core/language_models/base";

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
      temperature: 0.8,
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
    
    YOUR JOB: Until all the required information is gathered, you should ask the user very clear questions, guide them, and offer suggestions. STRICT rule: when we have gathered all the required information before starting the in-depth discussion, ask clearly the user for confirmation. Set "isInitialized" key to true only and only if the user confirm explicitly.

    REQUIRED INFORMATION:
    - discussion topic: this could be a question, a thesis to debate, a common belief to examine, a philosophical concept to explore, a thought experiment to conduct, or a personal experience to analyze, among other possibilities. The user can propose a topic themselves, but if he doesn't, make immediatly some suggestions.
    - two characters name (and eventually their role): the discussion will be between the user and two characters that the user must also choose. Once the theme is selected, if the user doesn't suggest two names themselves, you have to ask the user which philosophers they want to have a discussion with and recommend some philosophers. Since this is primarily meant to be a philosophical conversation, you will always suggest Socrates first, who will play the role of the (falsely) naive questioner, and another philosopher of our choice who could be relevant to the chosen theme and who will propose and defend specific theses or concepts. If the user rejects several proposals or explicitly requests it, you can suggest other types of real or fictional characters from other fields.
    - user name: if the user hasn't introduced themselves, you ll at least ask for their first name.

    OUTPUT: Each of our responses will be in JSON format, following the provided schema.`,
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
      conversationTopic: z.string(),
      userName: z.string(),
      firstCharacterName: z.string(),
      firstCharacterRole: z.string(),
      secondCharacterName: z.string(),
      secondCharacterRole: z.string(),
      isInitialized: z
        .boolean()
        .describe(
          "Are all requested infos (topic and two Characters names) fixed in preparation of the conversation ?",
        ),
    }),
  );

  //   const llm = new ChatOpenAI({
  //     model,
  //     temperature: 0.8,
  //   });
  //   const structuredLlm = llm.withStructuredOutput(
  //     z.object({
  //       message: z
  //         .string()
  //         .describe(
  //           "Your response messages to help the user to find a topic and two Character names",
  //         ),
  //       conversationTopic: z.string(),
  //       userName: z.string(),
  //       firstCharacterName: z.string(),
  //       firstCharacterRole: z.string(),
  //       secondCharacterName: z.string(),
  //       secondCharacterRole: z.string(),
  //       isInitialized: z
  //         .boolean()
  //         .describe(
  //           "Are all requested infos (topic and two Characters names) fixed in preparation of the conversation ?",
  //         ),
  //     }),
  //   );

  const response = await llm.invoke(allMessages);
  const { message, ...rest } = response;

  return {
    messages: [message], //[...state.messages, response.messages],
    initialization: {
      ...rest,
    },
  };
}

export async function roleDefiner(state: State, config?: RunnableConfig) {
  const systemMessage = `
    CONTEXT:
    The goal, based on what you're going to answer, is to create an engaging and thought-provoking debate: ${state.initialization.firstCharacterName} and ${state.initialization.secondCharacterName} will take part in a debate led by a powerful and rigorous AI like you.
    
    YOUR JOB:
    For each of them, you have to compose a detailed and effective prompt to guide an AI like you so that it plays their role as realistically and convincingly as possible. Their speech, way of expressing themselves, and especially their ideas, arguments, examples, or thought experiments should be as much as possible directly inspired by their major works. They should rely on concepts, arguments, examples, or thought experiments and phrases they actually used (rephrasing them as clearly and accessibly as possible for non-specialists). The proposed ideas must be consistent with their system of thought and style of thinking. Anachronisms (especially for examples) are quite possible as long as they provide an opportunity for a small humorous wink. Specify in broad terms what characterizes their style, character, and way of reasoning, which should be imitated. Since the discussion will be about ${state.initialization.conversationTopic}, indicate what important, strong, and original principles (at least 3 or 4) of their philosophy could be mobilized and with which they should be consistent.
    
    - Here is a simple example for Socrate: "As Socrate in Platon's dialogues, you adopt his ironic tone, you ask questions to encourage your interlocutor to clarify the meaning of the main concepts they use, forcing them to refine their definitions through amusing counter-examples or thought-provoking analogies. You're usually one step ahead in conversations. You know why you're asking certain innocent questions - to lead the other person into contradicting themselves or exposing their lack of knowledge."

    - Here's what is said about their role and character at this stage. Take it up, reflect on it and now complete and improve it so that each prompt is more precise and meets the criteria mentioned above and all the criteria you deem good for an AI to best follow these prompts. Make sure you don't oversimplify the character's thoughts with clichés, and leave room for deep and original reflections that could draw from lesser-known ideas of the author.
       - about ${state.initialization.firstCharacterName}'s role: ${state.initialization.firstCharacterRole}
       - about ${state.initialization.secondCharacterName}'s role: ${state.initialization.secondCharacterRole}

    The response will be in JSON format. Write your prompt for ${state.initialization.firstCharacterName} in the "firstCharacterRole" key, and for ${state.initialization.secondCharacterName} in the "secondCharacterRole" key.`;

  const structuredLlm = getLlm(
    config?.configurable?.moderatorModel,
    z.object({
      firstCharacterRole: z.string(),
      secondCharacterRole: z.string(),
    }),
  );

  const response = await structuredLlm.invoke(systemMessage);

  return {
    messages: [
      `Now we will discuss about the following topic: ${state.initialization.conversationTopic}`,
    ],
    initialization: {
      firstCharacterRole: response.firstCharacterRole,
      secondCharacterRole: response.secondCharacterRole,
    },
  };
}

export async function firstChatBotNode(
  state: State,
  config?: RunnableConfig,
): Promise<Update> {
  const systemMessage = {
    role: "system",
    content: `You're taking part in a lively discussion about ${state.initialization.conversationTopic}, playing the role of ${state.initialization.firstCharacterName} (and you play only it's role). Your main conversation partner is ${state.initialization.secondCharacterName} (but you don't play it's role).
    
    More precisely, you will play the following role: ${state.initialization.firstCharacterRole}.
        
    ${dialogInstructions}`,
  };

  const messages = [systemMessage, ...state.messages];

  const llm = getLlm(config?.configurable?.bot1Model);
  const response = await llm.invoke(messages);
  state.messages = [response];
  state.initialization.sourceSpeaker = "Bot1";
  return state;
}

export async function secondChatBotNode(state: State, config?: RunnableConfig) {
  const systemMessage = {
    role: "system",
    content: `You're taking part in a lively discussion about ${state.initialization.conversationTopic}, playing the role of ${state.initialization.secondCharacterName} (and you play only it's role). Your main conversation partner is ${state.initialization.firstCharacterName} (but you don't play it's role).
  
    More precisely, you will play the following role: ${state.initialization.secondCharacterRole}.

    If you want to stop the conversation with ${state.initialization.firstCharacterName} because you feel like we're going in circles, you must respond only with a single word: "FINISHED".

${dialogInstructions}`,
  };

  const llm = getLlm(config?.configurable?.bot2Model);

  const messages = [systemMessage, ...state.messages];
  //   await delay(5000)
  const response = await llm.invoke(messages);

  state.messages = [response];
  state.initialization.sourceSpeaker = "Bot2";
  return state;
  //   return {
  //     messages: [response],
  //     // initialization: state.initialization,
  //   };
}

export async function humanInput(state: State) {
  return state;
}

export async function moderation(state: State, config?: RunnableConfig) {
  const messages = state.messages;
  const lastMessage = messages.at(-1);
  let response;
  if (lastMessage?.constructor.name === "HumanMessage") {
    if (lastMessage.content === ">") state.messages.pop();
    else {
      const length = messages.length;
      let recentConversation: string = "Human: " + messages[length - 1];
      for (let i = 0; i < 3; i++) {
        if (i > length) break;
        recentConversation =
          "\n\n" + messages[length - 1 - i].content + recentConversation;
      }
      recentConversation.trim();

      const systemMessage = `
        CONTEXT:
        The goal, based on what you're going to answer, is to lead an engaging and thought-provoking debate about ${state.initialization.conversationTopic} between the user, namely ${state.initialization.userName}, and two characters: ${state.initialization.firstCharacterName} and ${state.initialization.secondCharacterName} (an AI is simulating their role).

        YOUR JOB:
        From the last contribution to the debate and the last response of the human user, determine which character the message written by the human is addressed to, or in other words, who should speak in the next round of the debate. Your response will be recorded in the JSON key "targetSpeaker": if it's ${state.initialization.firstCharacterName}, answer with "Bot1", if it's ${state.initialization.secondCharacterName}, answer with "Bot2"
        
        3 last contribution (at most) in the debate:
        ${recentConversation} 
        `;
      // , if it's ${state.initialization.userName}, answer with "Human"

      console.log(systemMessage);

      const structuredLlm = getLlm(
        config?.configurable?.moderatorModel,
        z.object({
          targetSpeaker: z.string().describe("Bot1|Bot2"),
        }),
      );

      response = await structuredLlm.invoke(systemMessage);

      console.log("response :>> ", response);
      response && (state.initialization.targetSpeaker = response.targetSpeaker);
      state.initialization.sourceSpeaker = "Human";
    }
  }

  return state;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function updateNodeMetadata(state: State) {
  const previousNode = state.metadata.currentNode || "start";
  state.metadata.previousNode = previousNode;
  state.metadata.currentNode = "myNode";
  return state;
}
