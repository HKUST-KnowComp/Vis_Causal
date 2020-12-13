import json
import torch
from pytorch_transformers import *
import logging
import argparse
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import math
import numpy
import collections
import random
from tqdm import tqdm



class DataLoader:
    def __init__(self, data_path, k):
        self.k = k
        with open(data_path, 'r') as f:
            raw_dataset = json.load(f)
        self.word_embeddings = self.load_embedding_dict('glove.txt') # glove path
        self.train_set = self.tensorize_example(raw_dataset['training'], 'train')
        print('successfully loaded %d examples for training data' % len(self.train_set))
        self.dev_set = self.tensorize_example(raw_dataset['validation'], 'dev')
        print('successfully loaded %d examples for dev data' % len(self.dev_set))

        self.test_set = self.tensorize_example(raw_dataset['testing'], 'test')
        print('successfully loaded %d examples for test data' % len(self.test_set))

    def load_embedding_dict(self, path):
        print("Loading word embeddings from {}...".format(path))
        default_embedding = numpy.zeros(300)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        if len(path) > 0:
            vocab_size = None
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    word_end = line.find(" ")
                    word = line[:word_end]
                    embedding = numpy.fromstring(line[word_end + 1:], numpy.float32, sep=" ")
                    assert len(embedding) == 300
                    embedding_dict[word] = embedding
            if vocab_size is not None:
                assert vocab_size == len(embedding_dict)
            print("Done loading word embeddings.")
        return embedding_dict

    def tensorize_example(self, initial_dataset_by_video, mode):
        if mode == 'test' or mode=='dev':
            tensorized_dataset = list()
            for tmp_video_id in initial_dataset_by_video:
                tmp_video = initial_dataset_by_video[tmp_video_id]
                # if event exists
                if len(tmp_video['image_0']['event']) > 0:
                    tensorized_dataset += self.test_tensorize_frame(tmp_video['image_0'], tmp_video['image_1'],
                                                               tmp_video['category'], tmp_video_id.split('_')[1], int(0))
                if len(tmp_video['image_1']['event']) > 0:
                    tensorized_dataset += self.test_tensorize_frame(tmp_video['image_1'], tmp_video['image_2'],
                                                               tmp_video['category'], tmp_video_id.split('_')[1], int(1))
                if len(tmp_video['image_2']['event']) > 0:
                    tensorized_dataset += self.test_tensorize_frame(tmp_video['image_2'], tmp_video['image_3'],
                                                               tmp_video['category'], tmp_video_id.split('_')[1], int(2))
                if len(tmp_video['image_3']['event']) > 0:
                    tensorized_dataset += self.test_tensorize_frame(tmp_video['image_3'], tmp_video['image_4'],
                                                               tmp_video['category'], tmp_video_id.split('_')[1], int(3))
        if mode == 'train':
            tensorized_dataset = list()
            for tmp_video_id in initial_dataset_by_video:
                tmp_video = initial_dataset_by_video[tmp_video_id]
                # if event exists
                if len(tmp_video['image_0']['event']) > 0:
                    tensorized_dataset += self.tensorize_frame(tmp_video['image_0'], tmp_video['image_1'],
                                                               tmp_video['category'], tmp_video_id.split('_')[1], int(0))
                if len(tmp_video['image_1']['event']) > 0:
                    tensorized_dataset += self.tensorize_frame(tmp_video['image_1'], tmp_video['image_2'],
                                                               tmp_video['category'], tmp_video_id.split('_')[1], int(1))
                if len(tmp_video['image_2']['event']) > 0:
                    tensorized_dataset += self.tensorize_frame(tmp_video['image_2'], tmp_video['image_3'],
                                                               tmp_video['category'], tmp_video_id.split('_')[1], int(2))
                if len(tmp_video['image_3']['event']) > 0:
                    tensorized_dataset += self.tensorize_frame(tmp_video['image_3'], tmp_video['image_4'],
                                                               tmp_video['category'], tmp_video_id.split('_')[1], int(3))
        return tensorized_dataset

    def test_tensorize_frame(self, image_one, image_two, category, video_id, image_id):
        all_entities = dict()
        for tmp_entity in image_one['entity']:
            if tmp_entity not in all_entities:
                all_entities[tmp_entity] = image_one['entity'][tmp_entity]
            else:
                all_entities[tmp_entity] += image_one['entity'][tmp_entity]
        for tmp_entity in image_two['entity']:
            if tmp_entity not in all_entities:
                all_entities[tmp_entity] = image_two['entity'][tmp_entity]
            else:
                all_entities[tmp_entity] += image_two['entity'][tmp_entity]
        sorted_entities = sorted(all_entities, key=lambda x: all_entities[x], reverse=True)
        sorted_entities = sorted_entities[:self.k]

        tensorized_entities = list()

        for tmp_entity in sorted_entities:
            tensorized_entities.append(self.word_embeddings[tmp_entity])
        tensorized_entities = torch.tensor(tensorized_entities).type(torch.float32).to(device)

        tensorized_examples_for_one_frame = list()
        for tail_dict in image_one['event'].values():
            for tmp_event_pair in tail_dict:
                event_1 = tmp_event_pair[0].split('$$')[0]
                event_2 = tmp_event_pair[0].split('$$')[1]
                event_1_embeddings = list()
                event_2_embeddings = list()
                for w in event_1.split(' '):
                    event_1_embeddings.append(self.word_embeddings[w.lower()])
                for w in event_2.split(' '):
                    event_2_embeddings.append(self.word_embeddings[w.lower()])
                if len(event_1_embeddings) > 1 and len(event_2_embeddings) > 1:
                    tensorized_event_1 = torch.tensor(event_1_embeddings).type(torch.float32).to(device)
                    tensorized_event_2 = torch.tensor(event_2_embeddings).type(torch.float32).to(device)
                    if args.model == 'ResNetAsContext':
                        image_one_vector = numpy.array(image_one['vector'])
                        image_two_vector = numpy.array(image_two['vector'])
                        overall_representation = image_one_vector + image_two_vector
                        overall_representation /= 2
                        resnet_representation = torch.tensor(overall_representation).type(torch.float32).to(device)
                    bert_tokenized_event_1 = tokenizer.encode('[CLS] ' + event_1 + ' . [SEP]')
                    bert_tokenized_event_2 = tokenizer.encode('[CLS] ' + event_2 + ' . [SEP]')

                    if args.model == 'ResNetAsContext':
                        tensorized_examples_for_one_frame.append({'event_1': tensorized_event_1,
                                                                  'event_2': tensorized_event_2,
                                                                  'bert_event_1': torch.tensor(bert_tokenized_event_1).to(
                                                                      device),
                                                                  'bert_event_2': torch.tensor(bert_tokenized_event_2).to(
                                                                      device),
                                                                  'entities': tensorized_entities,
                                                                  'label': torch.tensor([tmp_event_pair[1]]).to(device),
                                                                  'resnet_representation': resnet_representation,
                                                                  'category': category,
                                                                  'video_id': video_id,
                                                                  'image_id': image_id,
                                                                  'event_key': event_1})
                    else:
                        tensorized_examples_for_one_frame.append({'event_1': tensorized_event_1,
                                                                  'event_2': tensorized_event_2,
                                                                  'bert_event_1': torch.tensor(bert_tokenized_event_1).to(
                                                                      device),
                                                                  'bert_event_2': torch.tensor(bert_tokenized_event_2).to(
                                                                      device),
                                                                  'entities': tensorized_entities,
                                                                  'label': torch.tensor([tmp_event_pair[1]]).to(device),
                                                                  'category': category,
                                                                  'video_id': video_id,
                                                                  'image_id': image_id,
                                                                  'event_key': event_1})

        return tensorized_examples_for_one_frame

    def tensorize_frame(self, image_one, image_two, category, video_id, image_id):
        all_entities = dict()
        for tmp_entity in image_one['entity']:
            if tmp_entity not in all_entities:
                all_entities[tmp_entity] = image_one['entity'][tmp_entity]
            else:
                all_entities[tmp_entity] += image_one['entity'][tmp_entity]
        for tmp_entity in image_two['entity']:
            if tmp_entity not in all_entities:
                all_entities[tmp_entity] = image_two['entity'][tmp_entity]
            else:
                all_entities[tmp_entity] += image_two['entity'][tmp_entity]
        sorted_entities = sorted(all_entities, key=lambda x: all_entities[x], reverse=True)
        sorted_entities = sorted_entities[:self.k]

        tensorized_entities = list()

        for tmp_entity in sorted_entities:
            tensorized_entities.append(self.word_embeddings[tmp_entity])
        tensorized_entities = torch.tensor(tensorized_entities).type(torch.float32).to(device)

        tensorized_examples_for_one_frame = list()
        for tail_dict in image_one['event'].values():
            positive_list = list()
            negative_list = list()
            for tmp_event_pair in tail_dict:
                if tmp_event_pair[1] == 1:
                    positive_list.append(tmp_event_pair)
                if tmp_event_pair[1] == 0:
                    negative_list.append(tmp_event_pair)
            random.shuffle(negative_list)
            negative_list = negative_list[:len(positive_list)]
            positive_list.extend(negative_list)
            candidate_list = positive_list

            for tmp_event_pair in candidate_list:
                event_1 = tmp_event_pair[0].split('$$')[0]
                event_2 = tmp_event_pair[0].split('$$')[1]
                event_1_embeddings = list()
                event_2_embeddings = list()
                for w in event_1.split(' '):
                    event_1_embeddings.append(self.word_embeddings[w.lower()])
                for w in event_2.split(' '):
                    event_2_embeddings.append(self.word_embeddings[w.lower()])
                if len(event_1_embeddings) > 1 and len(event_2_embeddings) > 1:
                    tensorized_event_1 = torch.tensor(event_1_embeddings).type(torch.float32).to(device)
                    tensorized_event_2 = torch.tensor(event_2_embeddings).type(torch.float32).to(device)
                    if args.model == 'ResNetAsContext':
                        image_one_vector = numpy.array(image_one['vector'])
                        image_two_vector = numpy.array(image_two['vector'])
                        overall_representation = image_one_vector + image_two_vector
                        overall_representation /= 2
                        resnet_representation = torch.tensor(overall_representation).type(torch.float32).to(device)
                    bert_tokenized_event_1 = tokenizer.encode('[CLS] ' + event_1 + ' . [SEP]')
                    bert_tokenized_event_2 = tokenizer.encode('[CLS] ' + event_2 + ' . [SEP]')
                    if args.model == 'ResNetAsContext':
                        tensorized_examples_for_one_frame.append({'event_1':tensorized_event_1,
                                                                  'event_2':tensorized_event_2,
                                                                  'bert_event_1': torch.tensor(bert_tokenized_event_1).to(
                                                                      device),
                                                                  'bert_event_2': torch.tensor(bert_tokenized_event_2).to(
                                                                      device),
                                                                  'entities': tensorized_entities,
                                                                  'label': torch.tensor([tmp_event_pair[1]]).to(device),
                                                                  'resnet_representation': resnet_representation,
                                                                  'category': category,
                                                                  'video_id': video_id,
                                                                  'image_id': image_id,
                                                                  'event_key': event_1
                                                                  })
                    else:
                        tensorized_examples_for_one_frame.append({'event_1':tensorized_event_1,
                                                                  'event_2':tensorized_event_2,
                                                                  'bert_event_1': torch.tensor(bert_tokenized_event_1).to(
                                                                      device),
                                                                  'bert_event_2': torch.tensor(bert_tokenized_event_2).to(
                                                                      device),
                                                                  'entities': tensorized_entities,
                                                                  'label': torch.tensor([tmp_event_pair[1]]).to(device),
                                                                  'category': category,
                                                                  'video_id': video_id,
                                                                  'image_id': image_id,
                                                                  'event_key': event_1
                                                                  })


        return tensorized_examples_for_one_frame


class NoContext(BertModel):
    def __init__(self, config):

        super(BertModel, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_dim = 768


        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, event_1, event_2, entities):
        event_1_representation, _ = self.bert(event_1.unsqueeze(0)) #[1, seq_len, 768]
        event_1_representation = torch.mean(event_1_representation, dim=1) # [1, 768]
        event_2_representation, _ = self.bert(event_2.unsqueeze(0))
        event_2_representation = torch.mean(event_2_representation, dim=1)
        overall_representation = torch.cat([event_1_representation, event_2_representation], dim=1) # [1, 768*2]
        overall_representation = self.dropout(overall_representation)
        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction


class NoAttention(BertModel):
    def __init__(self, config):
        super(NoAttention, self).__init__(config)
        self.bert = BertModel(config)

        self.embedding_size = 300
        self.hidden_dim = 768

        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(self.hidden_dim * 2 + self.embedding_size, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, event_1, event_2, entities):
        event_1_representation, _ = self.bert(event_1.unsqueeze(0))
        event_1_representation = torch.mean(event_1_representation, dim=1) # [1, 768]
        event_2_representation, _ = self.bert(event_2.unsqueeze(0))
        event_2_representation = torch.mean(event_2_representation, dim=1)
        entity_representation = torch.mean(entities.unsqueeze(0), dim=1) # [1, 300]
        overall_representation = torch.cat([event_1_representation, event_2_representation, entity_representation],
                                           dim=1) #[1, 1836]
        overall_representation = self.dropout(overall_representation)
        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction

class ResNetAsContext(BertModel):
    def __init__(self, config):
        super(ResNetAsContext, self).__init__(config)
        self.bert = BertModel(config)

        self.hidden_dim = 768
        self.compress_size = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.compress = torch.nn.Linear(2048, self.compress_size)
        self.second_last_layer = torch.nn.Linear(self.hidden_dim * 2 + self.compress_size, self.hidden_dim)
        # self.second_last_layer = torch.nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.last_layer = torch.nn.Linear(self.hidden_dim, 2)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, event_1, event_2, resnet_representation):
        event_1_representation, _ = self.bert(event_1.unsqueeze(0))
        event_1_representation = torch.mean(event_1_representation, dim=1)
        event_2_representation, _ = self.bert(event_2.unsqueeze(0))
        event_2_representation = torch.mean(event_2_representation, dim=1)

        overall_representation = torch.cat([event_1_representation, event_2_representation, self.compress(resnet_representation.unsqueeze(0))], dim=1)


        prediction = self.last_layer(self.second_last_layer(overall_representation))
        return prediction

class BERTCausal(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.bert = BertModel(config)
        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(768*2 + self.embedding_size * 2, self.hidden_dim)

        self.attention_to_entity = torch.nn.Linear(768 + self.embedding_size, 1)
        self.attention_to_word = torch.nn.Linear(768 + self.embedding_size, 1)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def cross_attention(self, eventuality_representation, entity_representations):

        number_of_words = eventuality_representation.size(0)
        number_of_objects = entity_representations.size(0)
        event_raw_representation = torch.mean(eventuality_representation, dim=0)  # 768

        event_raw_representation = event_raw_representation.repeat(number_of_objects, 1)  # 10*768

        event_attention = self.attention_to_entity(
            torch.cat([event_raw_representation, entity_representations], dim=1))  # 10 * 1

        context_representation = torch.mean(entity_representations * event_attention.repeat(1, self.embedding_size),
                                            dim=0)  # 1 * 300


        context_representation_for_attention = context_representation.repeat(number_of_words, 1)


        word_attention = self.attention_to_word(
            torch.cat([eventuality_representation, context_representation_for_attention], dim=1))  # num_words * 1

        event_representation = torch.mean(eventuality_representation * word_attention.repeat(1, 768),
                                          dim=0)

        return event_representation.unsqueeze(0), context_representation.unsqueeze(0)

    def forward(self, event_1, event_2, entities):
        event_1_representation = self.bert(event_1.unsqueeze(0))
        event_1_weighted_representation, event_1_context_representation = self.cross_attention(
            event_1_representation[0].squeeze(),
            entities)
        event_2_representation = self.bert(event_2.unsqueeze(0))
        event_2_weighted_representation, event_2_context_representation = self.cross_attention(
            event_2_representation[0].squeeze(),
            entities)

        overall_representation = torch.cat(
            [event_1_weighted_representation, event_2_weighted_representation, event_1_context_representation,
             event_2_context_representation], dim=1)
        overall_representation = self.dropout(overall_representation)

        prediction = self.second_last_layer(overall_representation)
        return prediction



class GPTCausal(GPT2Model):
    def __init__(self, config):
        super(GPT2Model, self).__init__(config)
        self.lm = GPT2Model(config)
        self.embedding_size = 300
        self.hidden_dim = 200

        self.dropout = torch.nn.Dropout(0.5)
        self.second_last_layer = torch.nn.Linear(768*2 + self.embedding_size * 2, self.hidden_dim)


        self.attention_to_entity = torch.nn.Linear(768 + self.embedding_size, 1)
        self.attention_to_word = torch.nn.Linear(768 + self.embedding_size, 1)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def cross_attention(self, eventuality_representation, entity_representations):

        number_of_words = eventuality_representation.size(0)
        number_of_objects = entity_representations.size(0)
        event_raw_representation = torch.mean(eventuality_representation, dim=0)  # 768

        event_raw_representation = event_raw_representation.repeat(number_of_objects, 1)  # 10*768

        event_attention = self.attention_to_entity(
            torch.cat([event_raw_representation, entity_representations], dim=1))  # 10 * 1

        context_representation = torch.mean(entity_representations * event_attention.repeat(1, self.embedding_size),
                                            dim=0)  # 1 * 300


        context_representation_for_attention = context_representation.repeat(number_of_words, 1)


        word_attention = self.attention_to_word(
            torch.cat([eventuality_representation, context_representation_for_attention], dim=1))  # num_words * 1

        event_representation = torch.mean(eventuality_representation * word_attention.repeat(1, 768),
                                          dim=0)

        return event_representation.unsqueeze(0), context_representation.unsqueeze(0)

    def forward(self, event_1, event_2, entities):
        event_1_representation = self.lm(event_1.unsqueeze(0))
        event_1_weighted_representation, event_1_context_representation = self.cross_attention(
            event_1_representation[0].squeeze(),
            entities)
        event_2_representation = self.lm(event_2.unsqueeze(0))
        event_2_weighted_representation, event_2_context_representation = self.cross_attention(
            event_2_representation[0].squeeze(),
            entities)

        overall_representation = torch.cat(
            [event_1_weighted_representation, event_2_weighted_representation, event_1_context_representation,
             event_2_context_representation], dim=1)
        overall_representation = self.dropout(overall_representation)

        prediction = self.second_last_layer(overall_representation)
        return prediction


def train(model, data):
    all_loss = 0
    print('training:')
    random.shuffle(data)
    model.train()
    for tmp_example in tqdm(data):

        if args.model == 'ResNetAsContext':
            final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                        resnet_representation=tmp_example['resnet_representation'])
        else:
            final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                        entities=tmp_example['entities'])

        loss = loss_func(final_prediction, tmp_example['label'])
        test_optimizer.zero_grad()
        loss.backward()
        test_optimizer.step()
        all_loss += loss.item()
    print('current loss:', all_loss / len(data))


def test(model, data):

    recall_list = [1, 2, 3, 5, 10]
    model.eval()
    random.shuffle(data)

    #initialize
    prediction_dict = dict()
    for video in range(100):
        prediction_dict['video_'+str(video)] = dict()
        for image in range(4):
            prediction_dict['video_'+str(video)]['image_'+str(image)] = dict()
    gt_positive_example = 0
    pred_positive_example = dict()
    for top_k in recall_list:
        pred_positive_example['top'+str(top_k)] = 0


    for tmp_example in data:
        if args.model == 'ResNetAsContext':
            final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                        resnet_representation=tmp_example['resnet_representation'])
        else:
            final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                        entities=tmp_example['entities'])

        softmax_prediction = F.softmax(final_prediction, dim = 1)

        tmp_one_result = dict()
        tmp_one_result['True_score'] = softmax_prediction.data[0][1]
        tmp_one_result['label'] = tmp_example['label'].item()

        if tmp_example['event_key'] not in prediction_dict['video_'+str(tmp_example['video_id'])]['image_'+str(tmp_example['image_id'])].keys():
            prediction_dict['video_'+str(tmp_example['video_id'])]['image_'+str(tmp_example['image_id'])][tmp_example['event_key']] = list()
        prediction_dict['video_' + str(tmp_example['video_id'])]['image_' + str(tmp_example['image_id'])][tmp_example['event_key']].append(tmp_one_result)

        if tmp_example['label'].data[0] == 1:
            gt_positive_example += 1

    for video in range(100):
        for image in range(4):
            current_predict = prediction_dict['video_'+str(video)]['image_'+str(image)]
            for key in current_predict:
                current_predict[key] = sorted(current_predict[key], key=lambda x: (x.get('True_score', 0)), reverse=True)
                # print(current_predict[key])
                for top_k in recall_list:
                    tmp_top_predict = current_predict[key][:top_k]
                    for tmp_example in tmp_top_predict:
                        if tmp_example['label'] == 1:
                            pred_positive_example['top' + str(top_k)] += 1

    recall_result = dict()
    for top_k in recall_list:
        recall_result['Recall_' + str(top_k)] = pred_positive_example['top' + str(top_k)] / gt_positive_example

    # return correct_count / len(data)
    return recall_result


def test_by_type(model, data, recall_k):
    correct_count = dict()
    all_count = dict()
    correct_count['overall'] = 0
    all_count['overall'] = 0
    model.eval()
    random.shuffle(data)

    # initialize
    prediction_dict = dict()
    for video in range(100):
        prediction_dict['video_' + str(video)] = dict()
        for image in range(4):
            prediction_dict['video_' + str(video)]['image_' + str(image)] = dict()

    for tmp_example in data:
        if args.model == 'ResNetAsContext':
            final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                        resnet_representation=tmp_example['resnet_representation'])
        else:
            final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                        entities=tmp_example['entities'])


        softmax_prediction = F.softmax(final_prediction, dim=1)

        if tmp_example['category'] not in correct_count:
            correct_count[tmp_example['category']] = 0
        if tmp_example['category'] not in all_count:
            all_count[tmp_example['category']] = 0

        tmp_one_result = dict()
        tmp_one_result['True_score'] = softmax_prediction.data[0][1]
        tmp_one_result['label'] = tmp_example['label'].item()
        tmp_one_result['category'] = tmp_example['category']

        if tmp_example['event_key'] not in prediction_dict['video_' + str(tmp_example['video_id'])][
            'image_' + str(tmp_example['image_id'])].keys():
            prediction_dict['video_' + str(tmp_example['video_id'])]['image_' + str(tmp_example['image_id'])][
                tmp_example['event_key']] = list()
        prediction_dict['video_' + str(tmp_example['video_id'])]['image_' + str(tmp_example['image_id'])][
            tmp_example['event_key']].append(tmp_one_result)

        if tmp_example['label'].data[0] == 1:
            all_count['overall'] += 1
            all_count[tmp_example['category']] += 1

    for video in range(100):
        for image in range(4):
            current_predict = prediction_dict['video_' + str(video)]['image_' + str(image)]
            for key in current_predict:
                current_predict[key] = sorted(current_predict[key], key=lambda x: (x.get('True_score', 0)),
                                              reverse=True)
                # print(current_predict[key])
                tmp_top_predict = current_predict[key][:recall_k]
                for tmp_example in tmp_top_predict:
                    if tmp_example['label'] == 1:
                        correct_count[tmp_example['category']] += 1
                        correct_count['overall'] += 1

    accuracy_by_type = dict()
    for tmp_category in all_count:
        accuracy_by_type[tmp_category] = correct_count[tmp_category] / all_count[tmp_category]

    return accuracy_by_type



parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--gpu", default='0', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--k", default=10, type=int, required=False,
                    help="choose top k entities")
parser.add_argument("--model", default='Full-Model', type=str, required=False,
                    help="choose the model to test")
parser.add_argument("--lr", default=0.0001, type=float, required=False,
                    help="initial learning rate")
parser.add_argument("--lrdecay", default=0.98, type=float, required=False,
                    help="learning rate decay every 5 epochs")
parser.add_argument("--test_by_type", default=False, required=False,
                    help="Evaluate the model by types (i.e., Sports, Socializing, Household, Personal Care, Eating)")

args = parser.parse_args()

# Use gpu
logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device:', device)
n_gpu = torch.cuda.device_count()
print('number of gpu:', n_gpu)
torch.cuda.get_device_name(0)


if args.model == 'Full-Model':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    current_model = BERTCausal.from_pretrained('bert-base-uncased')
elif args.model == 'NoContext':
    current_model = NoContext.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif args.model == 'NoAttention':
    current_model = NoAttention.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif args.model == 'ResNetAsContext':
    current_model = ResNetAsContext.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif args.model == 'GPT-2':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    current_model = GPTCausal.from_pretrained('gpt2')
else:
    print('Please Choose one model!!')
print(args.model)

dataset_path = 'Context_dataset_wo_vector.json'
all_data = DataLoader(dataset_path, args.k)
current_model.to(device)
test_optimizer = torch.optim.SGD(current_model.parameters(), lr=args.lr)
loss_func = torch.nn.CrossEntropyLoss()


best_dev_performance = dict()
final_performance = dict()
if args.test_by_type:
    accuracy_by_type = dict()
for top_k in [1, 2, 3, 5, 10]:
    best_dev_performance['Recall_'+str(top_k)] = 0
    final_performance['Best_'+str(top_k)] = dict()
    if args.test_by_type:
        accuracy_by_type['Recall_'+str(top_k)] = dict()

# ten epoch
for i in range(10):
    print('Iteration:', i+1, '|', 'Current best performance:', final_performance)
    train(current_model, all_data.train_set)

    dev_performance = test(current_model, all_data.dev_set)
    print('Dev accuracy:', dev_performance)

    test_performance = test(current_model, all_data.test_set)
    print('Test accuracy:', test_performance)

    for top_k in [1, 2, 3, 5, 10]:
        if dev_performance['Recall_' + str(top_k)] > best_dev_performance['Recall_' + str(top_k)]:
            print('Recall@'+str(top_k)+': New best performance!!!')
            best_dev_performance['Recall_' + str(top_k)] = dev_performance['Recall_' + str(top_k)]
            final_performance['Best_' + str(top_k)] = test_performance
            if args.test_by_type:
                accuracy_by_type['Recall_'+str(top_k)] = test_by_type(current_model, all_data.test_set, top_k)
                print(accuracy_by_type)
            print('Recall@' + str(top_k) + ': We are saving the new best model!')
            torch.save(current_model.state_dict(), 'models/' + 'Recall' + str(top_k) + '_' + args.model + '.pth')

if args.test_by_type:
    print(accuracy_by_type)
print(final_performance)
print('End.')

