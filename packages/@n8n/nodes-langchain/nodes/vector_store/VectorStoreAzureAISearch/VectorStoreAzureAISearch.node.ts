import { AzureKeyCredential, SearchIndexClient } from '@azure/search-documents';
import {
	AzureAISearchVectorStore,
	AzureAISearchQueryType,
} from '@langchain/community/vectorstores/azure_aisearch';
import { Document } from '@langchain/core/documents';
import type { EmbeddingsInterface } from '@langchain/core/embeddings';
import {
	NodeOperationError,
	type IDataObject,
	type ILoadOptionsFunctions,
	type INodeProperties,
	type IExecuteFunctions,
	type ISupplyDataFunctions,
} from 'n8n-workflow';

import { createVectorStoreNode } from '../shared/createVectorStoreNode/createVectorStoreNode';

// User agent for usage tracking
const USER_AGENT_PREFIX = 'n8n-azure-ai-search';

// Knowledge Base API version (preview)
const KB_API_VERSION = '2025-11-01-preview';

export const AZURE_AI_SEARCH_CREDENTIALS = 'azureAiSearchApi';
export const INDEX_NAME = 'indexName';
export const QUERY_TYPE = 'queryType';
export const FILTER = 'filter';
export const SEMANTIC_CONFIGURATION = 'semanticConfiguration';

// Knowledge Base specific constants
export const RETRIEVAL_MODE = 'retrievalMode';
export const KNOWLEDGE_BASE_NAME = 'knowledgeBaseName';
export const OUTPUT_MODE = 'outputMode';
export const REASONING_EFFORT = 'reasoningEffort';
export const INCLUDE_ACTIVITY = 'includeActivity';

const indexNameField: INodeProperties = {
	displayName: 'Index Name',
	name: INDEX_NAME,
	type: 'string',
	default: 'n8n-vectorstore',
	description:
		'The name of the Azure AI Search index. Will be created automatically if it does not exist.',
	required: true,
	displayOptions: {
		hide: {
			[RETRIEVAL_MODE]: ['knowledgeBase'],
		},
	},
};

const queryTypeField: INodeProperties = {
	displayName: 'Query Type',
	name: QUERY_TYPE,
	type: 'options',
	default: 'hybrid',
	description: 'The type of search query to perform',
	options: [
		{
			name: 'Vector',
			value: 'vector',
			description: 'Vector similarity search only',
		},
		{
			name: 'Hybrid',
			value: 'hybrid',
			description: 'Combines vector and keyword search (recommended)',
		},
		{
			name: 'Semantic Hybrid',
			value: 'semanticHybrid',
			description: 'Hybrid search with semantic ranking (requires Basic tier or higher)',
		},
	],
};

const filterField: INodeProperties = {
	displayName: 'Filter',
	name: FILTER,
	type: 'string',
	default: '',
	description:
		'Filter results using OData syntax. Use metadata/fieldName for metadata fields. <a href="https://learn.microsoft.com/en-us/azure/search/search-query-odata-filter" target="_blank">Learn more</a>.',
	placeholder: "metadata/category eq 'technology' and metadata/author eq 'John'",
};

const semanticConfigurationField: INodeProperties = {
	displayName: 'Semantic Configuration',
	name: SEMANTIC_CONFIGURATION,
	type: 'string',
	default: '',
	description: 'Name of the semantic configuration for semantic ranking (optional)',
	displayOptions: {
		show: {
			[QUERY_TYPE]: ['semanticHybrid'],
		},
	},
};

// Retrieval Mode selector - top-level choice between Index and Knowledge Base
const retrievalModeField: INodeProperties = {
	displayName: 'Retrieval Mode',
	name: RETRIEVAL_MODE,
	type: 'options',
	noDataExpression: true,
	default: 'index',
	description: 'Choose between a search index or knowledge base for retrieval',
	options: [
		{
			name: 'Search Index',
			value: 'index',
			description: 'Use a traditional Azure AI Search index',
		},
		{
			name: 'Knowledge Base (Preview)',
			value: 'knowledgeBase',
			description: 'Use FoundryIQ Knowledge Base for agentic retrieval with LLM-powered answers',
		},
	],
	displayOptions: {
		show: {
			mode: ['load', 'retrieve', 'retrieve-as-tool'],
		},
	},
};

// Knowledge Base specific fields
const knowledgeBaseNameField: INodeProperties = {
	displayName: 'Knowledge Base Name',
	name: KNOWLEDGE_BASE_NAME,
	type: 'string',
	required: true,
	default: '',
	description: 'Name of the Knowledge Base to query',
	displayOptions: {
		show: {
			[RETRIEVAL_MODE]: ['knowledgeBase'],
		},
	},
};

const outputModeField: INodeProperties = {
	displayName: 'Output Mode',
	name: OUTPUT_MODE,
	type: 'options',
	default: 'answerSynthesis',
	description: 'How to format retrieval results',
	options: [
		{
			name: 'Answer Synthesis',
			value: 'answerSynthesis',
			description: 'LLM-generated answer with citations',
		},
		{
			name: 'Extracted Data',
			value: 'extractedData',
			description: 'Raw search results from knowledge sources',
		},
	],
	displayOptions: {
		show: {
			[RETRIEVAL_MODE]: ['knowledgeBase'],
		},
	},
};

const reasoningEffortField: INodeProperties = {
	displayName: 'Reasoning Effort',
	name: REASONING_EFFORT,
	type: 'options',
	default: 'low',
	description: 'How much LLM processing to use for query planning',
	options: [
		{
			name: 'Minimal',
			value: 'minimal',
			description: 'No LLM - queries all sources',
		},
		{
			name: 'Low',
			value: 'low',
			description: 'Basic LLM source selection (default)',
		},
		{
			name: 'Medium',
			value: 'medium',
			description: 'More LLM processing for relevance',
		},
	],
	displayOptions: {
		show: {
			[RETRIEVAL_MODE]: ['knowledgeBase'],
		},
	},
};

const includeActivityField: INodeProperties = {
	displayName: 'Include Activity',
	name: INCLUDE_ACTIVITY,
	type: 'boolean',
	default: false,
	description: 'Whether to include retrieval activity details in response',
	displayOptions: {
		show: {
			[RETRIEVAL_MODE]: ['knowledgeBase'],
		},
	},
};

// Notice for KB mode about embeddings
const kbEmbeddingsNotice: INodeProperties = {
	displayName:
		'Knowledge Base mode handles embeddings internally. The Embedding input connection is not used.',
	name: 'kbEmbeddingsNotice',
	type: 'notice',
	default: '',
	displayOptions: {
		show: {
			[RETRIEVAL_MODE]: ['knowledgeBase'],
		},
	},
};

const sharedFields: INodeProperties[] = [
	retrievalModeField,
	kbEmbeddingsNotice,
	knowledgeBaseNameField,
	outputModeField,
	reasoningEffortField,
	includeActivityField,
	indexNameField,
];

const retrieveFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [queryTypeField, filterField, semanticConfigurationField],
		displayOptions: {
			hide: {
				[RETRIEVAL_MODE]: ['knowledgeBase'],
			},
		},
	},
];

const insertFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [
			{
				displayName: 'Clear Index',
				name: 'clearIndex',
				type: 'boolean',
				default: false,
				description:
					'Whether to delete and recreate the index before inserting new data. Warning: This will reset any custom index configuration (semantic ranking, analyzers, etc.) to defaults.',
			},
		],
	},
];

type IFunctionsContext = IExecuteFunctions | ISupplyDataFunctions | ILoadOptionsFunctions;

function isExecutionContext(
	context: IFunctionsContext,
): context is IExecuteFunctions | ISupplyDataFunctions {
	// IExecuteFunctions and ISupplyDataFunctions have addInputData method
	// ILoadOptionsFunctions does not
	return 'addInputData' in context;
}

function getParameter(key: string, context: IFunctionsContext, itemIndex: number): string {
	let value: unknown;

	if (isExecutionContext(context)) {
		// Execution context: includes itemIndex parameter
		value = context.getNodeParameter(key, itemIndex, '', { extractValue: true });
	} else {
		// Load options context: no itemIndex parameter
		value = context.getNodeParameter(key, '', { extractValue: true });
	}

	if (typeof value !== 'string') {
		throw new NodeOperationError(context.getNode(), `Parameter ${key} must be a string`);
	}
	return value;
}

export const getIndexName = getParameter.bind(null, INDEX_NAME);

function getOptionValue<T>(
	name: string,
	context: IExecuteFunctions | ISupplyDataFunctions,
	itemIndex: number,
	defaultValue?: T,
): T | undefined {
	const options: IDataObject = context.getNodeParameter('options', itemIndex, {});
	return options[name] !== undefined ? (options[name] as T) : defaultValue;
}

interface ValidatedCredentials {
	endpoint: string;
	apiKey: string;
}

async function getValidatedCredentials(
	context: IFunctionsContext,
	itemIndex: number,
): Promise<ValidatedCredentials> {
	const credentials = await context.getCredentials(AZURE_AI_SEARCH_CREDENTIALS);

	if (!credentials.endpoint || typeof credentials.endpoint !== 'string') {
		throw new NodeOperationError(
			context.getNode(),
			'Azure AI Search endpoint is missing or invalid',
			{ itemIndex },
		);
	}

	if (!credentials.apiKey || typeof credentials.apiKey !== 'string') {
		throw new NodeOperationError(context.getNode(), 'API Key is required for authentication', {
			itemIndex,
		});
	}

	return {
		endpoint: credentials.endpoint,
		apiKey: credentials.apiKey,
	};
}

/**
 * Deletes an Azure AI Search index if clearIndex option is enabled.
 * Exported for testing purposes.
 */
export async function clearAzureSearchIndex(
	context: IFunctionsContext,
	itemIndex: number,
): Promise<boolean> {
	const options = context.getNodeParameter('options', itemIndex, {}) as {
		clearIndex?: boolean;
	};

	if (!options.clearIndex) {
		return false;
	}

	const credentials = await getValidatedCredentials(context, itemIndex);
	const indexName = getIndexName(context, itemIndex);

	try {
		const indexClient = new SearchIndexClient(
			credentials.endpoint,
			new AzureKeyCredential(credentials.apiKey),
		);
		await indexClient.deleteIndex(indexName);
		context.logger.debug(`Deleted Azure AI Search index: ${indexName}`);
		return true;
	} catch (deleteError) {
		// Log the error but don't fail - index might not exist yet
		context.logger.debug('Error deleting index (may not exist):', {
			message: deleteError instanceof Error ? deleteError.message : String(deleteError),
		});
		return false;
	}
}

async function getAzureAISearchClient(
	context: IFunctionsContext,
	embeddings: EmbeddingsInterface,
	itemIndex: number,
): Promise<AzureAISearchVectorStore> {
	const credentials = await getValidatedCredentials(context, itemIndex);

	try {
		const indexName = getIndexName(context, itemIndex);
		const azureCredentials = new AzureKeyCredential(credentials.apiKey);

		// Pass endpoint, indexName, and credentials to enable automatic index creation
		// LangChain will create the index automatically if it doesn't exist
		const config: any = {
			endpoint: credentials.endpoint,
			indexName,
			credentials: azureCredentials,
			search: {},
			// Add custom user agent for usage tracking
			clientOptions: {
				userAgentOptions: { userAgentPrefix: USER_AGENT_PREFIX },
			},
		};

		// Set search configuration options only for execution contexts
		if (isExecutionContext(context)) {
			const queryType = getQueryType(context, itemIndex);
			const semanticConfiguration = getOptionValue<string>(
				'semanticConfiguration',
				context,
				itemIndex,
			);
			const filter = getOptionValue<string>('filter', context, itemIndex);

			config.search.type = queryType;

			if (filter) {
				config.search.filter = filter;
			}

			if (queryType === AzureAISearchQueryType.SemanticHybrid && semanticConfiguration) {
				config.search.semanticConfigurationName = semanticConfiguration;
			}
		}

		return new AzureAISearchVectorStore(embeddings, config);
	} catch (error) {
		if (error instanceof NodeOperationError) {
			throw error;
		}

		// Log the full error for debugging
		context.logger.debug('Azure AI Search connection error:', {
			message: error instanceof Error ? error.message : String(error),
			code: (error as any).code,
			statusCode: (error as any).statusCode,
			details: (error as any).details,
		});

		// Check for authentication errors
		if (
			error.message?.includes('401') ||
			error.message?.includes('Unauthorized') ||
			error.message?.includes('authentication failed')
		) {
			throw new NodeOperationError(
				context.getNode(),
				'Authentication failed - invalid API key or endpoint.',
				{
					itemIndex,
					description:
						'Please verify your API Key and Search Endpoint are correct in the credentials configuration.',
				},
			);
		}

		// Check for authorization errors (403)
		if (error.message?.includes('403') || error.message?.includes('Forbidden')) {
			throw new NodeOperationError(
				context.getNode(),
				'Authorization failed - insufficient permissions.',
				{
					itemIndex,
					description:
						'The API Key does not have sufficient permissions. Ensure the key has the required access level for this operation.',
				},
			);
		}

		const errorMessage = error instanceof Error ? error.message : String(error);
		throw new NodeOperationError(context.getNode(), `Error: ${errorMessage}`, {
			itemIndex,
			description: 'Please check your Azure AI Search connection details',
		});
	}
}

function getQueryType(
	context: IExecuteFunctions | ISupplyDataFunctions,
	itemIndex: number,
): AzureAISearchQueryType {
	const queryType = getOptionValue<string>('queryType', context, itemIndex, 'hybrid');

	switch (queryType) {
		case 'vector':
			return AzureAISearchQueryType.Similarity;
		case 'hybrid':
			return AzureAISearchQueryType.SimilarityHybrid;
		case 'semanticHybrid':
			return AzureAISearchQueryType.SemanticHybrid;
		default:
			return AzureAISearchQueryType.SimilarityHybrid;
	}
}

// ============================================================================
// Knowledge Base Types and Functions
// ============================================================================

interface KnowledgeBaseRetrievalRequest {
	messages: Array<{
		role: 'system' | 'assistant' | 'user';
		content: Array<{ type: 'text'; text: string }>;
	}>;
	includeActivity?: boolean;
}

interface KnowledgeBaseRetrievalResponse {
	response: Array<{
		content: Array<{ type: 'text'; text: string }>;
	}>;
	activity?: unknown;
}

/**
 * Retrieves data from a Knowledge Base using the Azure AI Search Agentic Retrieval API.
 * This is a direct REST API call since LangChain.JS does not support Knowledge Bases.
 */
async function retrieveFromKnowledgeBase(
	context: IExecuteFunctions | ISupplyDataFunctions,
	itemIndex: number,
	query: string,
): Promise<KnowledgeBaseRetrievalResponse> {
	const credentials = await getValidatedCredentials(context, itemIndex);
	const knowledgeBaseName = context.getNodeParameter(KNOWLEDGE_BASE_NAME, itemIndex, '') as string;
	const includeActivity = context.getNodeParameter(INCLUDE_ACTIVITY, itemIndex, false) as boolean;

	if (!knowledgeBaseName) {
		throw new NodeOperationError(
			context.getNode(),
			'Knowledge Base Name is required when using Knowledge Base mode',
			{ itemIndex },
		);
	}

	const url = `${credentials.endpoint}/knowledgebases('${encodeURIComponent(knowledgeBaseName)}')/retrieve?api-version=${KB_API_VERSION}`;

	const requestBody: KnowledgeBaseRetrievalRequest = {
		messages: [
			{
				role: 'user',
				content: [{ type: 'text', text: query }],
			},
		],
		includeActivity,
	};

	try {
		const response = await context.helpers.httpRequest({
			method: 'POST',
			url,
			headers: {
				'Content-Type': 'application/json',
				'api-key': credentials.apiKey,
				Accept: 'application/json;odata.metadata=minimal',
			},
			body: requestBody,
			json: true,
		});

		return response as KnowledgeBaseRetrievalResponse;
	} catch (error) {
		const errorObj = error as { message?: string; statusCode?: number };
		const errorMessage = error instanceof Error ? error.message : String(error);

		// Log the full error for debugging
		context.logger.debug('Knowledge Base retrieval error:', {
			message: errorMessage,
			statusCode: errorObj.statusCode,
		});

		// Check for authentication errors
		if (
			errorMessage.includes('401') ||
			errorMessage.includes('Unauthorized') ||
			errorObj.statusCode === 401
		) {
			throw new NodeOperationError(
				context.getNode(),
				'Authentication failed - invalid API key or endpoint.',
				{
					itemIndex,
					description:
						'Please verify your API Key and Search Endpoint are correct in the credentials configuration.',
				},
			);
		}

		// Check for not found errors (KB doesn't exist)
		if (errorMessage.includes('404') || errorObj.statusCode === 404) {
			throw new NodeOperationError(
				context.getNode(),
				`Knowledge Base '${knowledgeBaseName}' not found.`,
				{
					itemIndex,
					description:
						'Please verify the Knowledge Base name is correct and exists in your Azure AI Search service.',
				},
			);
		}

		throw new NodeOperationError(
			context.getNode(),
			`Knowledge Base retrieval error: ${errorMessage}`,
			{
				itemIndex,
				description: 'Please check your Knowledge Base configuration and Azure AI Search service.',
			},
		);
	}
}

/**
 * Converts a Knowledge Base retrieval response to LangChain Document format.
 */
function kbResponseToDocuments(response: KnowledgeBaseRetrievalResponse): Document[] {
	const documents: Document[] = [];

	for (const item of response.response) {
		for (const content of item.content) {
			if (content.type === 'text') {
				documents.push(
					new Document({
						pageContent: content.text,
						metadata: {
							source: 'knowledge-base',
						},
					}),
				);
			}
		}
	}

	return documents;
}

/**
 * A wrapper class that implements VectorStore-like interface for Knowledge Base retrieval.
 * This allows KB to be used with the existing n8n vector store infrastructure.
 */
class KnowledgeBaseVectorStoreWrapper {
	constructor(
		private context: IExecuteFunctions | ISupplyDataFunctions,
		private itemIndex: number,
	) {}

	async similaritySearch(query: string, k?: number): Promise<Document[]> {
		const response = await retrieveFromKnowledgeBase(this.context, this.itemIndex, query);
		return kbResponseToDocuments(response).slice(0, k ?? 4);
	}

	async similaritySearchWithScore(query: string, k?: number): Promise<Array<[Document, number]>> {
		const docs = await this.similaritySearch(query, k);
		// KB doesn't return scores, use 1.0 as placeholder for compatibility
		return docs.map((doc) => [doc, 1.0] as [Document, number]);
	}

	async similaritySearchVectorWithScore(
		_query: number[],
		_k: number,
		_filter?: unknown,
	): Promise<Array<[Document, number]>> {
		// For KB mode, we can't do vector search directly - we need the text query
		// This method is called by the base node, so we throw an informative error
		throw new NodeOperationError(
			this.context.getNode(),
			'Knowledge Base mode does not support direct vector search. Please use text-based search.',
			{ itemIndex: this.itemIndex },
		);
	}

	asRetriever(options?: { k?: number }) {
		const k = options?.k ?? 4;
		return {
			invoke: async (query: string) => await this.similaritySearch(query, k),
			getRelevantDocuments: async (query: string) => await this.similaritySearch(query, k),
		};
	}
}

/**
 * Gets the retrieval mode from node parameters.
 * Returns 'index' by default for backward compatibility.
 */
function getRetrievalMode(
	context: IExecuteFunctions | ISupplyDataFunctions,
	itemIndex: number,
): 'index' | 'knowledgeBase' {
	try {
		const mode = context.getNodeParameter(RETRIEVAL_MODE, itemIndex, 'index') as string;
		return mode === 'knowledgeBase' ? 'knowledgeBase' : 'index';
	} catch {
		// Parameter might not exist in older workflows
		return 'index';
	}
}

export class VectorStoreAzureAISearch extends createVectorStoreNode({
	meta: {
		displayName: 'Azure AI Search Vector Store',
		name: 'vectorStoreAzureAISearch',
		description: 'Work with your data in Azure AI Search Vector Store',
		icon: { light: 'file:azure-aisearch.svg', dark: 'file:azure-aisearch.svg' },
		docsUrl:
			'https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstoreazureaisearch/',
		credentials: [
			{
				name: 'azureAiSearchApi',
				required: true,
			},
		],
		operationModes: ['load', 'insert', 'retrieve', 'update', 'retrieve-as-tool'],
	},
	sharedFields,
	retrieveFields,
	loadFields: retrieveFields,
	insertFields,
	async getVectorStoreClient(context, _filter, embeddings, itemIndex) {
		// Check if we're in Knowledge Base mode (only for execution contexts)
		if (isExecutionContext(context)) {
			const retrievalMode = getRetrievalMode(context, itemIndex);

			if (retrievalMode === 'knowledgeBase') {
				// Return Knowledge Base wrapper instead of traditional vector store
				// The KB wrapper handles retrieval via direct REST API calls
				return new KnowledgeBaseVectorStoreWrapper(context, itemIndex) as any;
			}
		}

		// Index mode: use traditional Azure AI Search vector store
		const vectorStore = await getAzureAISearchClient(context, embeddings, itemIndex);

		// Apply OData filter to search methods if specified in options
		if (isExecutionContext(context)) {
			const filter = getOptionValue<string>('filter', context, itemIndex);

			if (filter) {
				// Per LangChain docs, pass filter as 3rd parameter with filterExpression
				const filterObject = { filterExpression: filter };

				// Override similaritySearchVectorWithScore - this is the method called by n8n base node
				const originalSearchVectorWithScore =
					vectorStore.similaritySearchVectorWithScore.bind(vectorStore);
				vectorStore.similaritySearchVectorWithScore = async (
					query: number[],
					k: number,
					additionalFilter?: any,
				) => {
					// Merge our OData filter with any additional filter passed by the caller
					const mergedFilter = additionalFilter
						? { ...filterObject, ...additionalFilter }
						: filterObject;
					return await originalSearchVectorWithScore(query, k, mergedFilter);
				};

				// Override similaritySearch to pass filter as 3rd parameter
				const originalSearch = vectorStore.similaritySearch.bind(vectorStore);
				vectorStore.similaritySearch = async (query: string, k?: number) => {
					return await originalSearch(query, k, filterObject);
				};

				// Override similaritySearchWithScore to pass filter as 3rd parameter
				const originalSearchWithScore = vectorStore.similaritySearchWithScore.bind(vectorStore);
				vectorStore.similaritySearchWithScore = async (query: string, k?: number) => {
					return await originalSearchWithScore(query, k, filterObject);
				};

				// Override asRetriever to inject filter into retriever options
				const originalAsRetriever = vectorStore.asRetriever.bind(vectorStore);
				vectorStore.asRetriever = (kwargs?: any) => {
					return originalAsRetriever({
						...kwargs,
						filter: filterObject,
					});
				};
			}
		}

		return vectorStore;
	},
	async populateVectorStore(context, embeddings, documents, itemIndex) {
		try {
			// Clear the index if requested (delete and recreate)
			await clearAzureSearchIndex(context, itemIndex);

			// Get vector store client (will auto-create index if it doesn't exist)
			const vectorStore = await getAzureAISearchClient(context, embeddings, itemIndex);

			// Add documents to Azure AI Search (framework handles batching)
			await vectorStore.addDocuments(documents);
		} catch (error) {
			// Log the full error for debugging
			context.logger.debug('Azure AI Search error details:', {
				message: error instanceof Error ? error.message : String(error),
				code: (error as any).code,
				statusCode: (error as any).statusCode,
				details: (error as any).details,
				stack: error instanceof Error ? error.stack : undefined,
			});

			// Check for authentication errors
			if (
				error.message?.includes('401') ||
				error.message?.includes('Unauthorized') ||
				error.message?.includes('authentication failed')
			) {
				throw new NodeOperationError(
					context.getNode(),
					'Authentication failed during document upload - invalid API key or endpoint.',
					{
						itemIndex,
						description:
							'Please verify your API Key and Search Endpoint are correct in the credentials configuration.',
					},
				);
			}

			// Check for authorization errors
			if (
				error.message?.includes('403') ||
				error.message?.includes('Forbidden') ||
				(error as any).statusCode === 403
			) {
				throw new NodeOperationError(
					context.getNode(),
					'Authorization failed - insufficient permissions for document upload.',
					{
						itemIndex,
						description:
							'The API Key does not have sufficient permissions for write operations. Ensure the key has the required access level.',
					},
				);
			}

			// Check for RestError (common Azure SDK error)
			if ((error as any).name === 'RestError' || error.message?.includes('RestError')) {
				const statusCode = (error as any).statusCode || 'unknown';
				const errorCode = (error as any).code || 'unknown';
				const errorMessage = error instanceof Error ? error.message : String(error);
				throw new NodeOperationError(
					context.getNode(),
					`Azure AI Search API error (${statusCode}): ${errorMessage}`,
					{
						itemIndex,
						description: `Error code: ${errorCode}\n\nCommon causes:\n- Invalid endpoint URL\n- Index doesn't exist\n- Authentication/authorization issues\n- API version mismatch\n\nCheck the console logs for detailed error information.`,
					},
				);
			}

			const errorMessage = error instanceof Error ? error.message : String(error);
			throw new NodeOperationError(context.getNode(), `Error: ${errorMessage}`, {
				itemIndex,
				description: 'Please check your Azure AI Search connection details and index configuration',
			});
		}
	},
}) {}
