import torch
import torch.nn as nn
import torch.functional as F

from torch.autograd import Variable
import numpy as np

def to_var(x, volatile=False):
   if torch.cuda.is_available():
       x = x.cuda()
   return Variable(x, volatile=volatile)

class Decoder(nn.Module):

    def __init__(self, vis_dim=512, vis_num=36, embed_dim=39, hidden_dim=512, vocab_size=39, num_layers=1, dropout_ratio=0.5):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.vis_dim = vis_dim
        self.vis_num = vis_num
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.A = 150
        self.B = 70
        self.N = 64

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.lstm_cell = nn.LSTMCell(embed_dim + vis_dim, hidden_dim, num_layers)
        self.fc_dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # # attention
        # self.att_vw = nn.Linear(self.vis_dim, self.vis_dim, bias=False)
        # self.att_hw = nn.Linear(self.hidden_dim, self.vis_dim, bias=False)
        # self.att_bias = nn.Parameter(torch.zeros(vis_num))
        # self.att_w = nn.Linear(self.vis_dim, 1, bias=False)

        # # decoder

        # # self.dec_image = decode_image()
        # #self.dec_image = nn.Linear(hidden_dim,self.N*self.N)
        # self.dec_image = DecoderConv(0.5)
        # self.dec_linear = nn.Linear(hidden_dim,5)
        # # self.dec_image = G_net()

    # def _attention_layer(self, features, hiddens):
    #     """
    #     :param features:  batch_size  * 196 * 512
    #     :param hiddens:  batch_size * hidden_dim
    #     :return:
    #     """
    #     att_fea = self.att_vw(features)
    #     # N-L-D
    #     att_h = self.att_hw(hiddens).unsqueeze(1)
    #     # N-1-D
    #     att_full = nn.ReLU()(att_fea + att_h + self.att_bias.view(1, -1, 1))
    #     att_out = self.att_w(att_full).squeeze(2)
    #     alpha = torch.softmax(att_out, dim=1)
    #     #print(alpha.shape)
    #     # N-L
    #     context = torch.mean(features * alpha.unsqueeze(2), 1)
    #     return context, alpha

    def forward(self, features, captions, lengths):
        """
        :param features: batch_size * 196 * 512
        :param captions: batch_size * time_steps
        :param lengths:
        :return:
        """
        batch_size, time_step, _ = features.shape
        # print('batch_size initialize')
        # print(batch_size)
        self.batch_size = batch_size
        # print(batch_size)
        # print(time_step)
        vocab_size = self.vocab_size
        embed = self.embed
        dropout = self.dropout
        attention_layer = self._attention_layer
        lstm_cell = self.lstm_cell
        fc_dropout = self.fc_dropout
        fc_out = self.fc_out
        caption_lengths, sort_ind = lengths.sort(dim=0, descending=True)
        captions = captions[sort_ind]
        features = features[sort_ind]

        word_embeddings = embed(captions)
        # print(word_embeddings.shape)
        word_embeddings = dropout(word_embeddings) if dropout is not None else word_embeddings
        feas = torch.mean(features, 1)  # batch_size * 512
        h, c = self.get_start_states(batch_size)

        # predicts = to_var(torch.zeros(batch_size, time_step, vocab_size))
        predictions = to_var(torch.zeros(batch_size, max(lengths), vocab_size))
        alphas = to_var(torch.zeros(batch_size, max(lengths), features.shape[1]))

        # for step in range(time_step):
        #     batch_size = sum(i >= step for i in lengths)
        #     if step != 0:
        #         feas, alpha = attention_layer(features[:batch_size, :], h0[:batch_size, :])
        #     words = (word_embeddings[:batch_size, step, :]).squeeze(1)
        #     inputs = torch.cat([feas, words], 1)
        #     h0, c0 = lstm_cell(inputs, (h0[:batch_size, :], c0[:batch_size, :]))
        #     outputs = fc_out(fc_dropout(h0)) if fc_dropout is not None else fc_out(h0)
        #     predicts[:batch_size, step, :] = outputs
        # print(lengths)
        lengths_t = (caption_lengths - 1).tolist()
        h_all =[]

        for t in range(max(lengths_t)):
            # print(t)
            # c_prev = Variable(torch.zeros(batch_size, 70*150)) if t == 0 else cs
            #batch_size_t = sum([l > t for l in lengths])
            #print(batch_size_t)
            attention_weighted_encoding, alpha = attention_layer(features, h)
            # print(attention_weighted_encoding.shape)
            # gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            # attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = lstm_cell(
                torch.cat([word_embeddings[:, t, :], attention_weighted_encoding], dim=1),
                (h, c))
            # print(h.shape)
            # concat hidden vectors at different steps
            # Pass through BIG GAN
            # cs = c_prev + self.write(fc_dropout(h), alpha)
            
            preds = fc_out(fc_dropout(h)) if fc_dropout is not None else fc_out(h)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha
            h_all.append(h)

        # call big gan here
        cs = self.dec_image(torch.cat(h_all,dim=0).reshape(16,-1,512))

        return predictions, captions, lengths_t, alphas, cs

    # def write(self, h_dec=0, alpha=None):
    #     w = self.dec_image(h_dec)
    #     w = w.view(self.batch_size, 64, 64)
    #     # w = Variable(torch.ones(4,5,5) * 3)
    #     # self.batch_size = 4
    #     (Fx, Fy), gamma = self.attn_window(h_dec)
    #     Fyt = Fy.transpose(2, 1)
    #     # wr = matmul(Fyt,matmul(w,Fx))
    #     wr = Fyt.bmm(w.bmm(Fx))
    #     wr = wr.view(self.batch_size, self.A * self.B)
    #     return wr / gamma.view(-1, 1).expand_as(wr)
    
    # def attn_window(self,h_dec):
    #     params = self.dec_linear(h_dec)
    #     gx_,gy_,log_sigma_2,log_delta,log_gamma = params.split(1,1)  #21

    #     # gx_ = Variable(torch.ones(4,1))
    #     # gy_ = Variable(torch.ones(4, 1) * 2)
    #     # log_sigma_2 = Variable(torch.ones(4, 1) * 3)
    #     # log_delta = Variable(torch.ones(4, 1) * 4)
    #     # log_gamma = Variable(torch.ones(4, 1) * 5)

    #     gx = (self.A + 1) / 2 * (gx_ + 1)    # 22
    #     gy = (self.B + 1) / 2 * (gy_ + 1)    # 23
    #     delta = (max(self.A,self.B) - 1) / (self.N - 1) * torch.exp(log_delta)  # 24
    #     sigma2 = torch.exp(log_sigma_2)
    #     gamma = torch.exp(log_gamma)

    #     return self.filterbank(gx,gy,sigma2,delta),gamma
    
    # def compute_mu(self,g,rng,delta):
    #     rng_t,delta_t = align(rng,delta)
    #     tmp = (rng_t - self.N / 2 - 0.5) * delta_t
    #     tmp_t,g_t = align(tmp,g)
    #     mu = tmp_t + g_t
    #     return mu

    # def filterbank(self,gx,gy,sigma2,delta):
    #     rng = Variable(torch.arange(0,self.N).view(1,-1))
    #     mu_x = self.compute_mu(gx,rng,delta)
    #     mu_y = self.compute_mu(gy,rng,delta)

    #     a = Variable(torch.arange(0,self.A).view(1,1,-1))
    #     b = Variable(torch.arange(0,self.B).view(1,1,-1))

    #     mu_x = mu_x.view(-1,self.N,1)
    #     mu_y = mu_y.view(-1,self.N,1)
    #     sigma2 = sigma2.view(-1,1,1)

    #     Fx = self.filterbank_matrices(a,mu_x,sigma2)
    #     Fy = self.filterbank_matrices(b,mu_y,sigma2)

    #     return Fx,Fy
    
    # def filterbank_matrices(self,a,mu_x,sigma2,epsilon=1e-9):
    #     t_a,t_mu_x = align(a,mu_x)
    #     temp = t_a - t_mu_x
    #     temp,t_sigma = align(temp,sigma2)
    #     temp = temp / (t_sigma * 2)
    #     F = torch.exp(-torch.pow(temp,2))
    #     F = F / (F.sum(2,True).expand_as(F) + epsilon)
    #     return F

    # def sample(self, feature, max_len=15):
    #     # greedy sample
    #     embed = self.embed
    #     lstm_cell = self.lstm_cell
    #     fc_out = self.fc_out
    #     attend = self._attention_layer
    #     batch_size = feature.size(0)

    #     sampled_ids = []
    #     alphas = [0]

    #     words = embed(to_var(torch.zeros(batch_size, 1).long())).squeeze(1)
    #     h0, c0 = self.get_start_states(batch_size)
    #     feas = torch.mean(feature, 1) # convert to batch_size*512

    #     for step in range(max_len):
    #         # if step != 0:
    #         feas, alpha = attend(feature, h0)
    #         alphas.append(alpha)
    #         inputs = torch.cat([words, feas], 1)
    #         h0, c0 = lstm_cell(inputs, (h0, c0))
    #         outputs = fc_out(h0)
    #         outputs = torch.softmax(outputs, dim=1)
    #         # print(outputs.shape)
    #         predicted = outputs.max(1)[1]
    #         # print(predicted.shape)
    #         sampled_ids.append(predicted.unsqueeze(1))
    #         words = embed(predicted)

    #     sampled_ids = torch.cat(sampled_ids, 1)
    #     return sampled_ids.squeeze(), alphas

    # def get_start_states(self, batch_size):
    #     hidden_dim = self.hidden_dim
    #     h0 = to_var(torch.zeros(batch_size, hidden_dim))
    #     c0 = to_var(torch.zeros(batch_size, hidden_dim))
    #     return h0, c0

    # def gen_sample(tparams, f_init, f_next, ctx0, options,
    #                trng=None, k=1, maxlen=30, stochastic=False, alpha=0.0, trie=None):
    #     """Generate captions with beam search.
    #     This function uses the beam search algorithm to conditionally
    #     generate candidate captions. Supports beamsearch and stochastic
    #     sampling.
    #     Parameters
    #     ----------
    #     tparams : OrderedDict()
    #         dictionary of theano shared variables represented weight
    #         matricies
    #     f_init : theano function
    #         input: annotation, output: initial lstm state and memory
    #         (also performs transformation on ctx0 if using lstm_encoder)
    #     f_next: theano function
    #         takes the previous word/state/memory + ctx0 and runs one
    #         step through the lstm
    #     ctx0 : numpy array
    #         annotation from convnet, of dimension #annotations x # dimension
    #         [e.g (196 x 512)]
    #     options : dict
    #         dictionary of flags and options
    #     trng : random number generator
    #     k : int
    #         size of beam search
    #     maxlen : int
    #         maximum allowed caption size
    #     stochastic : bool
    #         if True, sample stochastically
    #     Returns
    #     -------
    #     sample : list of list
    #         each sublist contains an (encoded) sample from the model
    #     sample_score : numpy array
    #         scores of each sample
    #     """
    #     if k > 1:
    #         assert not stochastic, 'Beam search does not support stochastic sampling'

    #     sample = []
    #     sample_score = []
    #     if stochastic:
    #         sample_score = 0

    #     live_k = 1
    #     dead_k = 0

    #     hyp_samples = [[]] * live_k
    #     hyp_scores = np.zeros(live_k).astype('float32')
    #     hyp_states = []
    #     hyp_memories = []

    #     # only matters if we use lstm encoder
    #     rval = f_init(ctx0)
    #     ctx0 = rval[0]
    #     next_state = []
    #     next_memory = []
    #     # the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    #     for lidx in xrange(options['n_layers_lstm']):
    #         next_state.append(rval[1 + lidx])
    #         next_state[-1] = next_state[-1].reshape([1, next_state[-1].shape[0]])
    #     for lidx in xrange(options['n_layers_lstm']):
    #         next_memory.append(rval[1 + options['n_layers_lstm'] + lidx])
    #         next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].shape[0]])
    #     # reminder: if next_w = -1, the switch statement
    #     # in build_sampler is triggered -> (empty word embeddings)
    #     next_w = -1 * numpy.ones((1,)).astype('int64')

    #     for ii in xrange(maxlen):
    #         # our "next" state/memory in our previous step is now our "initial" state and memory
    #         rval = f_next(*([next_w, ctx0] + next_state + next_memory))
    #         next_p = rval[0]
    #         next_w = rval[1]
    #         if trie != None:
    #             next_langP = get_lang_prob(hyp_samples, trie)

    #             # extract all the states and memories
    #         next_state = []
    #         next_memory = []
    #         for lidx in xrange(options['n_layers_lstm']):
    #             next_state.append(rval[2 + lidx])
    #             next_memory.append(rval[2 + options['n_layers_lstm'] + lidx])

    #         if stochastic:
    #             sample.append(next_w[0])  # if we are using stochastic sampling this easy
    #             sample_score += next_p[0, next_w[0]]
    #             if next_w[0] == 0:
    #                 break
    #         else:

    #             if trie != None:
    #                 # next_langP[:,0]=next_p[:,0]
    #                 idxmin = [idx for idx, x in enumerate(next_langP.flatten()) if x == sys.float_info.min]
    #                 next_p = next_p.flatten()
    #                 next_p[idxmin] = sys.float_info.min
    #                 next_p = next_p.reshape(-1, 38)
    #                 cand_scores = hyp_scores[:, None] - ((1 - alpha) * numpy.log(next_p)) - (
    #                             alpha * numpy.log(next_langP))  # add language probability
    #                 alpha = min(1 * alpha, 0.9)
    #             else:
    #                 cand_scores = hyp_scores[:, None] - numpy.log(next_p)
    #             cand_flat = cand_scores.flatten()
    #             ranks_flat = cand_flat.argsort()[:(k - dead_k)]  # (k-dead_k) numpy array of with min nll

    #             voc_size = next_p.shape[1]
    #             # indexing into the correct selected captions
    #             trans_indices = ranks_flat / voc_size
    #             word_indices = ranks_flat % voc_size
    #             costs = cand_flat[ranks_flat]  # extract costs from top hypothesis

    #             # a bunch of lists to hold future hypothesis
    #             new_hyp_samples = []
    #             new_hyp_scores = numpy.zeros(k - dead_k).astype('float32')
    #             new_hyp_states = []
    #             for lidx in xrange(options['n_layers_lstm']):
    #                 new_hyp_states.append([])
    #             new_hyp_memories = []
    #             for lidx in xrange(options['n_layers_lstm']):
    #                 new_hyp_memories.append([])

    #             # get the corresponding hypothesis and append the predicted word
    #             for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
    #                 new_hyp_samples.append(hyp_samples[ti] + [wi])
    #                 new_hyp_scores[idx] = copy.copy(costs[idx])  # copy in the cost of that hypothesis
    #                 for lidx in xrange(options['n_layers_lstm']):
    #                     new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
    #                 for lidx in xrange(options['n_layers_lstm']):
    #                     new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))

    #             # check the finished samples for <eos> character
    #             new_live_k = 0
    #             hyp_samples = []
    #             hyp_scores = []
    #             hyp_states = []
    #             for lidx in xrange(options['n_layers_lstm']):
    #                 hyp_states.append([])
    #             hyp_memories = []
    #             for lidx in xrange(options['n_layers_lstm']):
    #                 hyp_memories.append([])

    #             for idx in xrange(len(new_hyp_samples)):
    #                 if new_hyp_samples[idx][-1] == 0:
    #                     sample.append(new_hyp_samples[idx])
    #                     sample_score.append(new_hyp_scores[idx])
    #                     dead_k += 1  # completed sample!
    #                 else:
    #                     new_live_k += 1  # collect collect correct states/memories
    #                     hyp_samples.append(new_hyp_samples[idx])
    #                     hyp_scores.append(new_hyp_scores[idx])
    #                     for lidx in xrange(options['n_layers_lstm']):
    #                         hyp_states[lidx].append(new_hyp_states[lidx][idx])
    #                     for lidx in xrange(options['n_layers_lstm']):
    #                         hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
    #             hyp_scores = numpy.array(hyp_scores)
    #             live_k = new_live_k

    #             if new_live_k < 1:
    #                 break
    #             if dead_k >= k:
    #                 break

    #             next_w = numpy.array([w[-1] for w in hyp_samples])
    #             next_state = []
    #             for lidx in xrange(options['n_layers_lstm']):
    #                 next_state.append(numpy.array(hyp_states[lidx]))
    #             next_memory = []
    #             for lidx in xrange(options['n_layers_lstm']):
    #                 next_memory.append(numpy.array(hyp_memories[lidx]))

    #     if not stochastic:
    #         # dump every remaining one
    #         if live_k > 0:
    #             for idx in xrange(live_k):
    #                 sample.append(hyp_samples[idx])
    #                 sample_score.append(hyp_scores[idx])

    #     return sample, sample_score