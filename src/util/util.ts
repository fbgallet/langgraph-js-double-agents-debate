export const createReadingDelay = async (
  text: string,
  wordsPerMinute: number = 350,
) => {
  const wordCount = wordsCount(text);
  const delayPerWord = (60 / wordsPerMinute) * 1000;
  const delayInMilliseconds = delayPerWord * wordCount;
  console.log("delay in sec :>> ", delayInMilliseconds / 1000);
  await new Promise((resolve) => setTimeout(resolve, delayInMilliseconds));
};

const wordsCount = (txt: string): number => {
  const words = txt.trim().split(/\s+/);
  console.log("words :>> ", words);
  return words.length || 0;
};
