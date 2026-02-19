
// Fix: Updated model constants and added missing parseDocumentToQuestions implementation.
import { GoogleGenAI, Type } from "@google/genai";
import { Subject, ExamType, Question, QuestionType, Difficulty } from "./types";
import { getLocalQuestions } from "./data/jee_dataset";

// Configuration Defaults - Updated to recommended Gemini 3 models
const getModelConfig = () => {
  if (typeof window === 'undefined') return {};
  try {
    return JSON.parse(localStorage.getItem('nexus_model_config') || '{}');
  } catch (e) { return {}; }
};

const config = getModelConfig();
// Using gemini-3-pro-preview for complex JEE tasks and gemini-3-flash-preview for analysis/vision
const MODEL_ID = config.modelId || config.genModel || "gemini-3-pro-preview"; 
const VISION_MODEL = config.visionModel || "gemini-3-flash-preview"; 
const ANALYSIS_MODEL = config.analysisModel || "gemini-3-flash-preview";

// Global Rate Limiter
let lastRequestTimestamp = 0;
const MIN_GLOBAL_REQUEST_INTERVAL = 1000;

const enforceGlobalThrottle = async () => {
    const now = Date.now();
    const timeSinceLast = now - lastRequestTimestamp;
    if (timeSinceLast < MIN_GLOBAL_REQUEST_INTERVAL) {
        await new Promise(resolve => setTimeout(resolve, MIN_GLOBAL_REQUEST_INTERVAL - timeSinceLast));
    }
    lastRequestTimestamp = Date.now();
};

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Smart Generation following SDK guidelines
 */
const safeGenerateContent = async (params: any, retries = 3): Promise<any> => {
    await enforceGlobalThrottle();
    
    // Always use the prescribed initialization with the environment API key
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

    try {
        // Accessing models.generateContent directly as required
        return await ai.models.generateContent({
            model: params.model, 
            contents: params.contents,
            config: params.config
        });
    } catch (e: any) {
        console.warn(`Generation Error:`, e.message || e);
        
        const isRateLimit = e.message?.includes('429') || e.status === 429 || e.message?.toLowerCase().includes('quota');
        
        if (retries > 0) {
             await delay(2000);
             return safeGenerateContent(params, retries - 1);
        }
        
        if (isRateLimit) throw new Error("QUOTA_EXCEEDED");
        throw e;
    }
};

const cleanAndParseJSON = (text: string) => {
  if (!text) return null;
  let cleanText = text.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim();
  const startIdx = Math.max(cleanText.indexOf('{'), cleanText.indexOf('['));
  const endIdx = Math.max(cleanText.lastIndexOf('}'), cleanText.lastIndexOf(']'));
  if (startIdx === -1 || endIdx === -1) return null;
  try {
    return JSON.parse(cleanText.substring(startIdx, endIdx + 1));
  } catch (e) {
    return null;
  }
};

const extractText = (response: any): string => {
    try {
        // Direct access to the .text property as per guidelines
        return response.text || '';
    } catch (e) { return ''; }
};

const questionSchema = {
  type: Type.ARRAY,
  items: {
    type: Type.OBJECT,
    properties: {
      subject: { type: Type.STRING },
      chapter: { type: Type.STRING },
      type: { type: Type.STRING },
      difficulty: { type: Type.STRING },
      statement: { type: Type.STRING },
      options: { type: Type.ARRAY, items: { type: Type.STRING } },
      correctAnswer: { type: Type.STRING },
      solution: { type: Type.STRING },
      explanation: { type: Type.STRING },
      concept: { type: Type.STRING },
      markingScheme: {
         type: Type.OBJECT,
         properties: { positive: { type: Type.INTEGER }, negative: { type: Type.INTEGER } }
      }
    },
    required: ["subject", "statement", "correctAnswer", "solution", "type"]
  }
};

/**
 * Parses question paper PDF or images using Gemini Vision
 */
export const parseDocumentToQuestions = async (qFile: File, sFile?: File): Promise<Question[]> => {
    const fileToPart = async (file: File) => {
        const base64 = await new Promise<string>((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve((reader.result as string).split(',')[1]);
            reader.readAsDataURL(file);
        });
        return {
            inlineData: {
                data: base64,
                mimeType: file.type
            }
        };
    };

    const qPart = await fileToPart(qFile);
    const parts: any[] = [qPart];

    if (sFile) {
        parts.push(await fileToPart(sFile));
    }

    parts.push({ text: "Extract all JEE questions from this document. Use the second document (solution key) if provided to fill in correct answers, solutions, and concepts. Output a JSON array matching the question schema." });

    const response = await safeGenerateContent({
        model: VISION_MODEL,
        contents: { parts },
        config: {
            responseMimeType: "application/json",
            responseSchema: questionSchema,
        }
    });

    const data = cleanAndParseJSON(extractText(response));
    if (!Array.isArray(data)) return [];

    return data.map((q: any, i: number) => ({
        id: `parsed-${Date.now()}-${i}`,
        subject: q.subject || 'General',
        chapter: q.chapter || 'General',
        type: q.type || (q.options?.length ? 'MCQ' : 'Numerical'),
        difficulty: q.difficulty || 'Hard',
        statement: q.statement,
        options: q.options || [],
        correctAnswer: q.correctAnswer,
        solution: q.solution || q.explanation,
        explanation: q.explanation || '',
        concept: q.concept || '',
        markingScheme: q.markingScheme || { positive: 4, negative: q.options?.length ? 1 : 0 }
    }));
};

export const generateJEEQuestions = async (subject: Subject, count: number, type: ExamType, chapters?: string[], difficulty?: string | Difficulty, topics?: string[], distribution?: { mcq: number, numerical: number }): Promise<Question[]> => {
  const isFullSyllabus = !chapters || chapters.length === 0;
  const topicFocus = isFullSyllabus ? "Full Syllabus Mixed" : `Chapters: ${chapters.join(', ')}`;
  
  const entropy = `Salt:${Date.now()}-${Math.random().toString(36).substring(7)}`;

  const prompt = `
    TASK: Generate ${count} HIGH-LEVEL JEE ${subject} questions for ${type}.
    UNIQUE_ID: ${entropy}.
    DISTRIBUTION: ${distribution?.mcq || Math.ceil(count * 0.8)} MCQs, ${distribution?.numerical || Math.floor(count * 0.2)} Numericals.
    FOCUS: ${topicFocus}.
    FORMAT: Use LaTeX for Math ($...$). 
    RESPONSE: Return ONLY a JSON Array matching the schema.
  `;

  try {
    const response = await safeGenerateContent({
      model: MODEL_ID,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: questionSchema,
      }
    });

    const data = cleanAndParseJSON(extractText(response));
    if (!Array.isArray(data)) throw new Error("INVALID_JSON_RESPONSE");

    return data.map((q: any, i: number) => ({
        id: `ai-${Date.now()}-${i}-${Math.random().toString(36).substring(5)}`,
        subject: q.subject || subject,
        chapter: q.chapter || 'General',
        type: q.type || (q.options?.length ? 'MCQ' : 'Numerical'),
        difficulty: q.difficulty || 'Hard',
        statement: q.statement,
        options: q.options || [],
        correctAnswer: q.correctAnswer,
        solution: q.solution || q.explanation,
        explanation: q.explanation || '',
        concept: q.concept || '',
        markingScheme: q.markingScheme || { positive: 4, negative: q.options?.length ? 1 : 0 }
    }));
  } catch (e: any) {
    console.error("Generation Error:", e.message);
    if (e.message === "QUOTA_EXCEEDED") {
        return getLocalQuestions(subject, count);
    }
    throw e;
  }
};

export const generateFullJEEDailyPaper = async (config: any): Promise<{ physics: Question[], chemistry: Question[], mathematics: Question[] }> => {
  const subjects = [Subject.Physics, Subject.Chemistry, Subject.Mathematics];
  const results: any = { physics: [], chemistry: [], mathematics: [] };

  for (const sub of subjects) {
    const subKey = sub.toLowerCase() as 'physics' | 'chemistry' | 'mathematics';
    const subCfg = config[subKey];
    results[subKey] = await generateJEEQuestions(sub, subCfg.mcq + subCfg.numerical, ExamType.Advanced, subCfg.chapters, Difficulty.Hard, [], subCfg);
    await delay(1000); 
  }

  return results;
};

export const getDeepAnalysis = async (result: any) => {
    try {
        const response = await safeGenerateContent({
            model: ANALYSIS_MODEL,
            contents: `Analyze JEE performance data: ${JSON.stringify(result).substring(0, 4000)}. Provide strategic advice.`
        });
        return extractText(response);
    } catch (e) { return "Analysis insight unavailable."; }
};
