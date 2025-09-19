import type { EmbeddingsInterface } from '@langchain/core/embeddings';
import {
	AzureAISearchVectorStore,
	AzureAISearchQueryType,
} from '@langchain/community/vectorstores/azure_aisearch';
import { DefaultAzureCredential } from '@azure/identity';
import {
	type IDataObject,
	type ILoadOptionsFunctions,
	NodeOperationError,
	type INodeProperties,
	type IExecuteFunctions,
	type ISupplyDataFunctions,
} from 'n8n-workflow';
import { metadataFilterField } from '@utils/sharedFields';

import { createVectorStoreNode } from '../shared/createVectorStoreNode/createVectorStoreNode';

export const AZURE_AI_SEARCH_CREDENTIALS = 'azureAiSearchApi';
export const INDEX_NAME = 'indexName';
export const QUERY_TYPE = 'queryType';
export const RESULTS_COUNT = 'resultsCount';
export const FILTER = 'filter';
export const SEMANTIC_CONFIGURATION = 'semanticConfiguration';

const indexNameField: INodeProperties = {
	displayName: 'Index Name',
	name: INDEX_NAME,
	type: 'string',
	default: '',
	description: 'The name of the Azure AI Search index',
	required: true,
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
			name: 'Keyword',
			value: 'keyword',
			description: 'Traditional keyword search only',
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

const resultsCountField: INodeProperties = {
	displayName: 'Results Count',
	name: RESULTS_COUNT,
	type: 'number',
	default: 50,
	description: 'Number of results to return (maximum depends on your service tier)',
	typeOptions: {
		minValue: 1,
		maxValue: 1000,
	},
};

const filterField: INodeProperties = {
	displayName: 'Filter',
	name: FILTER,
	type: 'string',
	default: '',
	description: 'OData filter expression to apply to the search query',
	placeholder: "category eq 'technology' and rating ge 4",
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

const sharedFields: INodeProperties[] = [indexNameField];

const retrieveFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [
			queryTypeField,
			resultsCountField,
			filterField,
			semanticConfigurationField,
			metadataFilterField,
		],
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
				description: 'Whether to clear all documents in the index before inserting new data',
			},
		],
	},
];

type IFunctionsContext = IExecuteFunctions | ISupplyDataFunctions | ILoadOptionsFunctions;

function getParameter(key: string, context: IFunctionsContext, itemIndex: number): string {
	const value = context.getNodeParameter(key, itemIndex, '', {
		extractValue: true,
	}) as string;
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

async function getAzureAISearchClient(
	context: IFunctionsContext,
	embeddings: EmbeddingsInterface,
	itemIndex: number,
): Promise<AzureAISearchVectorStore> {
	try {
		const credentials = await context.getCredentials(AZURE_AI_SEARCH_CREDENTIALS);
		const indexName = getIndexName(context, itemIndex);

		const endpoint = credentials.endpoint as string;
		const authType = credentials.authType as string;

		let config: any = {
			endpoint,
			indexName,
		};

		if (authType === 'apiKey') {
			config.key = credentials.apiKey as string;
		} else if (authType === 'managedIdentitySystem') {
			config.credentials = new DefaultAzureCredential();
		} else if (authType === 'managedIdentityUser') {
			const clientId = credentials.managedIdentityClientId as string;
			config.credentials = new DefaultAzureCredential({
				managedIdentityClientId: clientId,
			});
		} else {
			throw new NodeOperationError(
				context.getNode(),
				`Unsupported authentication type: ${authType}`,
				{ itemIndex },
			);
		}

		// Set search configuration options
		const queryType = getQueryType(context, itemIndex);
		const resultsCount = getOptionValue<number>('resultsCount', context, itemIndex, 50);
		const filter = getOptionValue<string>('filter', context, itemIndex);
		const semanticConfiguration = getOptionValue<string>(
			'semanticConfiguration',
			context,
			itemIndex,
		);

		config.search = {
			type: queryType,
		};

		if (filter) {
			config.search.filter = filter;
		}

		if (queryType === AzureAISearchQueryType.SemanticHybrid && semanticConfiguration) {
			config.search.semanticConfiguration = semanticConfiguration;
		}

		return new AzureAISearchVectorStore(embeddings, config);
	} catch (error) {
		if (error instanceof NodeOperationError) {
			throw error;
		}
		throw new NodeOperationError(context.getNode(), `Error: ${error.message}`, {
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
			return AzureAISearchQueryType.SimilaritySearch;
		case 'keyword':
			return AzureAISearchQueryType.FullText;
		case 'hybrid':
			return AzureAISearchQueryType.SimilarityHybrid;
		case 'semanticHybrid':
			return AzureAISearchQueryType.SemanticHybrid;
		default:
			return AzureAISearchQueryType.SimilarityHybrid;
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
		return await getAzureAISearchClient(context, embeddings, itemIndex);
	},
	async populateVectorStore(context, embeddings, documents, itemIndex) {
		try {
			const vectorStore = await getAzureAISearchClient(context, embeddings, itemIndex);

			const clearIndex = getOptionValue<boolean>('clearIndex', context, itemIndex, false);

			if (clearIndex) {
				try {
					await vectorStore.delete({ deleteAll: true });
				} catch (error) {
					context.logger.warn(`Could not clear index: ${error.message}`);
				}
			}

			await vectorStore.addDocuments(documents);
		} catch (error) {
			throw new NodeOperationError(context.getNode(), `Error: ${error.message}`, {
				itemIndex,
				description: 'Please check your Azure AI Search connection details and index configuration',
			});
		}
	},
}) {}
