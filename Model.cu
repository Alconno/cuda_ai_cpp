
#include "Model.h"
#include <unordered_map>
#include "kernels.h"

ModelConfig::ModelConfig(int vocab_size, int block_size, int n_embd, int n_layer, int n_head, bool bias,
                         dt dropout, dt emb_std, dt attn_std, dt mlp_std, dt ln_gamma, dt linear_std)
    : vocab_size(vocab_size), block_size(block_size), n_embd(n_embd), n_layer(n_layer), n_head(n_head), bias(bias),
      dropout(dropout), emb_std(emb_std), attn_std(attn_std), mlp_std(mlp_std), ln_gamma(ln_gamma), linear_std(linear_std) {}

ModelConfig::ModelConfig() {}




LayerNorm::LayerNorm(ModelConfig config, dt eps)
	: config(config), features(config.n_embd), eps(eps) {
	initialize_weights();
}

LayerNorm::LayerNorm() {}

void LayerNorm::initialize_weights() {
	tmem["gamma"] = Tensor({ (int)features }, config.ln_gamma, "ln_gamma", 1);
	tmem["beta"] = Tensor({ (int)features }, 0.0, "ln_beta", 1);
}

void LayerNorm::initialize_mem() {
	tmem["eps"] = Tensor({ 1 }, eps, "eps");

	tmem["mean"] = Tensor({}, 0.0, "mean");
	tmem["diff"] = Tensor({}, 0.0, "diff");
	tmem["sq_diff"] = Tensor({}, 0.0, "sq_diff");
	tmem["var"] = Tensor({}, 0.0, "var");
	tmem["std"] = Tensor({}, 0.0, "std");
	tmem["std_sq"] = Tensor({}, 0.0, "std_sq");
	tmem["normalized"] = Tensor({}, 0.0, "normalized");
	tmem["scaled"] = Tensor({}, 0.0, "scaled");
	tmem["out"] = Tensor({}, 0.0, "ln_out");

	if (global_cuda_enabled) {
		tmem["eps"].to_gpu();
	}
}

void LayerNorm::forward(Tensor& x) {
	initialize_mem();

	tmem["mean"].mean(x, true);
	tmem["diff"].sub(x, tmem["mean"]);
	tmem["sq_diff"].square(tmem["diff"]);
	tmem["var"].mean(tmem["sq_diff"], true);
	tmem["std"].add(tmem["var"], tmem["eps"]);
	tmem["std_sq"].sqrt(tmem["std"]);
	tmem["normalized"].div(tmem["diff"], tmem["std_sq"]);
	tmem["scaled"].mul(tmem["gamma"], tmem["normalized"]);
	tmem["out"].add(tmem["scaled"], tmem["beta"]);
}

void LayerNorm::update(const dt& lr) {
	int N = config.n_embd;

	if (global_cuda_enabled) {
		int threads = 512, blocks = (N + threads - 1) / threads;

		auto update_param = [&](Tensor& param) {
			sgd_step_kernel << <blocks, threads >> > (param.d_data, param.d_grad, lr, N);
			CHECK_CUDA(cudaGetLastError());
			};

		update_param(tmem["gamma"]);
		update_param(tmem["beta"]);
	}
	else {
		for (int i = 0; i < N; i++) {
			tmem["gamma"].data[i] -= lr * tmem["gamma"].grad[i];
			tmem["beta"].data[i] -= lr * tmem["beta"].grad[i];
			tmem["gamma"].grad[i] = 0.0;
			tmem["beta"].grad[i] = 0.0;
		}
	}
}

void LayerNorm::set_gamma(std::vector<dt>& data) { tmem["gamma"].data = data; }
void LayerNorm::set_beta(std::vector<dt>& data) { tmem["beta"].data = data; }

void LayerNorm::to_gpu() {
	tmem["gamma"].to_gpu();
	tmem["beta"].to_gpu();
}

void LayerNorm::to_cpu() {
	tmem["gamma"].to_cpu();
	tmem["beta"].to_cpu();
}





SelfAttention::SelfAttention(ModelConfig config)
	: config(config),
	c_attn(config.n_embd, config.n_embd * 3, false, config.attn_std, 0.0),
	c_proj(config.n_embd, config.n_embd, false, config.attn_std, 0.0) {
	assert(config.n_embd % config.n_head == 0 && "n_embd must be divisible by n_head");
}

SelfAttention::SelfAttention() {}

void SelfAttention::initialize_mem(int B, int T, int C, int n_head) {
	int head_dim = C / n_head;

	tmem["qkv"] = Tensor({}, "qkv");
	tmem["c_proj_out"] = Tensor({}, "c_proj_out");
	tmem["out"] = Tensor({}, "attn_final");
	tmem["v_out"] = Tensor({ B, T, C }, {}, "v_out");
	tmem["scores"] = Tensor({ B, n_head, T, T }, {}, "scores");
	tmem["exps"] = Tensor({ B, n_head, T, T }, {}, "exps");
	tmem["smax"] = Tensor({ B, n_head, T, T }, {}, "smax");
}

void SelfAttention::forward(Tensor& x) {
	int B = x.shape[0], T = config.block_size, C = config.n_embd, n_head = config.n_head;
	initialize_mem(B, T, C, n_head);

	c_attn.forward(x, tmem["qkv"]);
	tmem["qkv"].scaled_dot_product_attention(tmem);
	c_proj.forward(tmem["v_out"], tmem["c_proj_out"]);

	// Residual connection
	tmem["out"].add(tmem["c_proj_out"], x);
}

void SelfAttention::update(const dt& lr) {
	auto update_tensor = [&](Tensor& tensor) {
		int numel = tensor.numel();
		int threads = 512;
		int blocks = (numel + threads - 1) / threads;

		sgd_step_kernel << <blocks, threads >> > (tensor.d_data, tensor.d_grad, lr, numel);
		CHECK_CUDA(cudaGetLastError());
		};

	if (global_cuda_enabled) {
		update_tensor(c_attn.weights);
		update_tensor(c_proj.weights);

		if (c_attn.bias)
			update_tensor(c_attn.bias_v);
		if (c_proj.bias)
			update_tensor(c_proj.bias_v);
	}
	else {
		auto update_cpu = [&](Tensor& tensor) {
			for (int i = 0; i < tensor.numel(); i++) {
				tensor.data[i] -= lr * tensor.grad[i];
				tensor.grad[i] = 0.0;
			}
			};

		update_cpu(c_attn.weights);
		update_cpu(c_proj.weights);

		if (c_attn.bias)
			update_cpu(c_attn.bias_v);
		if (c_proj.bias)
			update_cpu(c_proj.bias_v);
	}
}

void SelfAttention::to_gpu() {
	c_attn.weights.to_gpu(); c_attn.weights.constant_data = true;
	c_proj.weights.to_gpu(); c_proj.weights.constant_data = true;

	if (c_attn.bias) {
		c_attn.bias_v.to_gpu();
		c_attn.bias_v.constant_data = true;
	}
	if (c_proj.bias) {
		c_proj.bias_v.to_gpu();
		c_proj.bias_v.constant_data = true;
	}
}

void SelfAttention::to_cpu() {
	c_attn.weights.to_cpu();
	c_proj.weights.to_cpu();

	if (c_attn.bias) c_attn.bias_v.to_cpu();
	if (c_proj.bias) c_proj.bias_v.to_cpu();
}





MLP::MLP(ModelConfig config)
	: config(config),
	mlp(config.n_embd, config.n_embd * 4, 1, config.mlp_std, 0.0),
	mlp_proj(config.n_embd * 4, config.n_embd, 1, config.mlp_std, 0.0) {}

MLP::MLP() {}

void MLP::initialize_mem(int B, int T, int C) {
	tmem["mlp"] = Tensor({}, "mlp");
	tmem["gelu"] = Tensor({}, "gelu");
	tmem["mlp_proj"] = Tensor({}, "mlp_proj");
	tmem["out"] = Tensor({}, "out");
}

void MLP::forward(Tensor& x) {
	int B = x.shape[0], T = config.block_size, C = config.n_embd;
	initialize_mem(B, T, C);

	mlp.forward(x, tmem["mlp"]);
	tmem["mlp"].gelu(tmem["gelu"], true);
	mlp_proj.forward(tmem["gelu"], tmem["mlp_proj"]);

	// Residual connection
	tmem["out"].add(tmem["mlp_proj"], x);
}

void MLP::update(const dt& lr) {
	if (global_cuda_enabled) {
		const int threads = 512;

		auto update_cuda = [&](Linear& layer) {
			int numel = layer.weights.numel();
			int blocks = (numel + threads - 1) / threads;

			sgd_step_kernel << <blocks, threads >> > (layer.weights.d_data, layer.weights.d_grad, lr, numel);
			CHECK_CUDA(cudaGetLastError());

			if (layer.bias) {
				int bnumel = layer.bias_v.numel();
				int bblocks = (bnumel + threads - 1) / threads;
				sgd_step_kernel << <bblocks, threads >> > (
					layer.bias_v.d_data,
					layer.bias_v.d_grad,
					lr,
					bnumel
					);
			}
			};

		update_cuda(mlp);
		update_cuda(mlp_proj);
	}
	else {
		auto update_cpu = [&](Linear& layer) {
			for (int i = 0; i < layer.weights.numel(); ++i) {
				layer.weights.data[i] -= lr * layer.weights.grad[i];
				layer.weights.grad[i] = 0.0;
			}

			if (layer.bias) {
				for (int i = 0; i < layer.bias_v.numel(); ++i) {
					layer.bias_v.data[i] -= lr * layer.bias_v.grad[i];
					layer.bias_v.grad[i] = 0.0;
				}
			}
			};

		update_cpu(mlp);
		update_cpu(mlp_proj);
	}
}

void MLP::to_gpu() {
	mlp.weights.to_gpu();
	mlp_proj.weights.to_gpu();
	mlp.weights.constant_data = true;
	mlp_proj.weights.constant_data = true;

	if (mlp.bias) {
		mlp.bias_v.to_gpu();
		mlp.bias_v.constant_data = true;
	}
	if (mlp_proj.bias) {
		mlp_proj.bias_v.to_gpu();
		mlp_proj.bias_v.constant_data = true;
	}
}

void MLP::to_cpu() {
	mlp.weights.to_cpu();
	mlp_proj.weights.to_cpu();

	if (mlp.bias)
		mlp.bias_v.to_cpu();

	if (mlp_proj.bias)
		mlp_proj.bias_v.to_cpu();
}




Block::Block(ModelConfig config) {
	layers["ln_0"] = std::make_shared<LayerNorm>(config);
	layers["ln_1"] = std::make_shared<LayerNorm>(config);
	layers["attn"] = std::make_shared<SelfAttention>(config);
	layers["mlp"] = std::make_shared<MLP>(config);

	ln_0 = dynamic_cast<LayerNorm*>(layers["ln_0"].get());
	ln_1 = dynamic_cast<LayerNorm*>(layers["ln_1"].get());
	attn = dynamic_cast<SelfAttention*>(layers["attn"].get());
	mlp = dynamic_cast<MLP*>(layers["mlp"].get());
}

void Block::forward(Tensor*& hidden) {
	ln_0->forward(*hidden);
	attn->forward(ln_0->tmem["out"]);
	mlp->forward(attn->tmem["out"]);
	ln_1->forward(mlp->tmem["out"]);

	hidden = &ln_1->tmem["out"];
}

void Block::update(const dt& lr) {
	ln_0->update(lr);
	attn->update(lr);
	mlp->update(lr);
	ln_1->update(lr);
}

void Block::to_gpu() {
	ln_0->to_gpu();
	attn->to_gpu();
	mlp->to_gpu();
	ln_1->to_gpu();
}

void Block::to_cpu() {
	ln_0->to_cpu();
	attn->to_cpu();
	mlp->to_cpu();
	ln_1->to_cpu();
}





LayerList::LayerList(ModelConfig config) {
	for (int i = 0; i < config.n_layer; ++i) {
		blocks.emplace_back(std::make_shared<Block>(config));
	}
}

void LayerList::to_gpu() {
	for (auto& block : blocks)
		block->to_gpu();
}

void LayerList::to_cpu() {
	for (auto& block : blocks)
		block->to_cpu();
}





Model::Model(ModelConfig config, bool cuda) {
	this->config = config;

	// Initialize transformer modules
	transformer["wte"] = std::make_shared<Embedding>(config.vocab_size, config.n_embd, config.emb_std);
	transformer["wpe"] = std::make_shared<Embedding>(config.block_size, config.n_embd, config.emb_std);
	transformer["dropout"] = std::make_shared<Dropout>(config.dropout);
	transformer["h"] = std::make_shared<LayerList>(config);
	transformer["ln_f"] = std::make_shared<LayerNorm>(config);

	// Output layer
	lm_head = std::make_shared<Linear>(config.n_embd, config.vocab_size, 0, config.linear_std, 0.0);

	// Load or create model checkpoint
	//load_model("first_model");

	cuda_enabled = cuda;
	//std::this_thread::sleep_for(std::chrono::seconds(1));
	//save_model("first_model");
}


std::pair<std::vector<int>, std::vector<int>> generate_random_samples(
	int batch_size, int seq_length, int min_value, int max_value) {

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(min_value, max_value);

	std::vector<int> x, y;
	x.reserve(batch_size * seq_length);
	y.reserve(batch_size * seq_length);

	for (int i = 0; i < batch_size; ++i) {
		int num = dis(gen);
		for (int j = 0; j < seq_length; ++j) {
			x.push_back(num + j);
			y.push_back(num + j + 1);
		}
	}

	return { x, y };
}


void Model::train(const int& max_steps, const Optimizer& opt, const int& B) {
	std::cout << "Starting training with " << this->count_parameters() << " parameters\n";

	global_cuda_enabled = cuda_enabled;
	if (cuda_enabled) this->to_gpu();

	const int T = config.block_size;
	const int C = config.n_embd;

	std::vector<int> pos(T);
	std::iota(pos.begin(), pos.end(), 0);

	for (int step = 0; step <= max_steps; ++step) {
		auto t0 = std::chrono::high_resolution_clock::now();

		// -------------------- Data Generation --------------------
		auto xy = generate_random_samples(B, config.block_size, 0, config.vocab_size / 2 - 1);
		std::vector<int> input = xy.first;
		std::vector<int> targets = xy.second;

		// -------------------- Embeddings --------------------
		Tensor wte({}, "wte");
		dynamic_cast<Embedding*>(transformer["wte"].get())->forward(input, wte, B, T);

		Tensor wpe({}, "wpe");
		dynamic_cast<Embedding*>(transformer["wpe"].get())->forward(pos, wpe, 1, T);

		Tensor emb_out({}, "emb_out");
		emb_out.add(wte, wpe);

		if (config.dropout > 0.0) {
			dynamic_cast<Dropout*>(transformer["dropout"].get())->forward(emb_out);
		}

		// -------------------- Transformer Blocks --------------------
		Tensor* hidden = &emb_out;
		for (auto& l : dynamic_cast<LayerList*>(transformer["h"].get())->blocks) {
			dynamic_cast<Block*>(l.get())->forward(hidden);
		}

		// -------------------- Final LN + LM Head --------------------
		auto* ln_f = dynamic_cast<LayerNorm*>(transformer["ln_f"].get());
		ln_f->forward(*hidden);

		Tensor logits({ B, T, config.vocab_size }, {}, "logits");
		lm_head->forward(ln_f->tmem["out"], logits);

		// -------------------- Loss --------------------
		Tensor logits_softmax_out({}, "logits_softmax_out");
		logits.softmax(logits_softmax_out);

		Tensor loss({}, "loss");
		logits_softmax_out.cross_entropy(targets, loss);

		if (0) this->print_predictions(logits_softmax_out, logits.shape, targets);

		// -------------------- Backward + Update --------------------

		dt lr = get_lr(step, opt) * B;

		auto b0 = std::chrono::high_resolution_clock::now();
		loss.grad = { 1.0 };
		loss.backward();
		auto b1 = std::chrono::high_resolution_clock::now();

		lm_head->update(lr);
		ln_f->update(lr);
		for (auto& l : dynamic_cast<LayerList*>(transformer["h"].get())->blocks) {
			dynamic_cast<Block*>(l.get())->update(lr);
		}
		dynamic_cast<Embedding*>(transformer["wpe"].get())->update(lr);
		dynamic_cast<Embedding*>(transformer["wte"].get())->update(lr);

		// -------------------- Stats --------------------
		auto t1 = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(t1 - t0).count();
		double backward_seconds = std::chrono::duration<double>(b1 - b0).count();

		std::cout << "Backward pass took: " << backward_seconds << std::endl;

		int total_tokens = B * T;
		double tokens_per_sec = total_tokens / seconds;

		std::cout << "step: " << step
			<< " | time: " << seconds << "s"
			<< " | lr: " << lr
			<< " | loss: " << loss
			<< " | tokens/sec: " << tokens_per_sec << "\n";

		// -------------------- Save --------------------
		if (step > 0 && step % 25 == 0) {
			this->to_cpu();
			save_model("first_model");
			this->to_gpu();
		}

		loss.free_all();
		std::cout << std::flush;
	}
}


void Model::print_predictions(Tensor& logits_softmax_out, const std::vector<int>& logits_shape, const std::vector<int>& targets) {
	logits_softmax_out.to_cpu();

	const int batch_size = logits_shape[0];
	const int time_steps = logits_shape[1];
	const int vocab_size = logits_shape[2];

	std::vector<std::vector<int>> predicted_tokens(batch_size, std::vector<int>(time_steps));

	// Argmax over vocabulary
	for (int b = 0; b < batch_size; ++b) {
		for (int t = 0; t < time_steps; ++t) {
			int base_idx = (b * time_steps + t) * vocab_size;

			int max_idx = 0;
			float max_val = logits_softmax_out.data[base_idx];

			for (int v = 1; v < vocab_size; ++v) {
				float val = logits_softmax_out.data[base_idx + v];
				if (val > max_val) {
					max_val = val;
					max_idx = v;
				}
			}
			predicted_tokens[b][t] = max_idx;
		}
	}

	// Output predictions
	std::cout << "predicted tokens:\n";
	for (const auto& seq : predicted_tokens) {
		for (int token : seq) {
			std::cout << token << " ";
		}
		std::cout << '\n';
	}

	std::cout << "\nexpected:\n";
	for (size_t i = 0; i < targets.size(); ++i) {
		if (i > time_steps && i % time_steps == 0) std::cout << '\n';
		std::cout << targets[i] << ", ";
	}
	std::cout << "\n";

	logits_softmax_out.to_gpu();
}



void Model::to_gpu() {
	std::cout << "Moving model to GPU\n";

	dynamic_cast<Embedding*>(transformer["wte"].get())->to_gpu();
	dynamic_cast<Embedding*>(transformer["wpe"].get())->to_gpu();
	dynamic_cast<LayerList*>(transformer["h"].get())->to_gpu();
	dynamic_cast<LayerNorm*>(transformer["ln_f"].get())->to_gpu();
	lm_head->to_gpu();

	lm_head->weights.constant_data = true;
}

void Model::to_cpu() {
	std::cout << "Moving model to CPU\n";

	dynamic_cast<Embedding*>(transformer["wte"].get())->to_cpu();
	dynamic_cast<Embedding*>(transformer["wpe"].get())->to_cpu();
	dynamic_cast<LayerList*>(transformer["h"].get())->to_cpu();
	dynamic_cast<LayerNorm*>(transformer["ln_f"].get())->to_cpu();
	lm_head->to_cpu();
}

void Model::save_model(std::string model_name) {
	std::ofstream file(model_name + ".csv");
	if (file.is_open()) {
		for (auto& pair : this->transformer) {
			const std::string& layer_name = pair.first;
			auto& layer = pair.second;

			auto& layer_typeid = typeid(*layer);
			if (layer_typeid == typeid(Linear) || layer_typeid == typeid(Embedding)) {
				file << ("transformer." + layer_name) + ":";
				for (const auto& wi : layer->weights.data)
					file << wi << ", ";
				file << "\n";
			}
			else if (layer_typeid == typeid(LayerList)) {
				int block_idx = 0;
				for (const auto& block : dynamic_cast<LayerList*>(layer.get())->blocks) {
					for (auto& inner_pair : block->layers) {
						const std::string& inner_name = inner_pair.first;
						auto& inner_layer = inner_pair.second;

						auto& inner_layer_typeid = typeid(*inner_layer);
						file << ("transformer.h.block" + std::to_string(block_idx) + "." + inner_name);

						if (inner_layer_typeid == typeid(LayerNorm)) {
							file << ".gamma:";
							for (const auto& wi : dynamic_cast<LayerNorm*>(inner_layer.get())->tmem["gamma"].data)
								file << wi << ", ";
							file << ("\ntransformer.h.block" + std::to_string(block_idx) + "." + inner_name) + ".beta:";
							for (const auto& wi : dynamic_cast<LayerNorm*>(inner_layer.get())->tmem["beta"].data)
								file << wi << ", ";
						}
						else if (inner_layer_typeid == typeid(SelfAttention)) {
							auto attn_layer = dynamic_cast<SelfAttention*>(inner_layer.get());
							file << ".c_attn:";
							for (const auto& wi : attn_layer->c_attn.get_weights().data)
								file << wi << ", ";
							file << ("\ntransformer.h.block" + std::to_string(block_idx) + "." + inner_name + ".c_proj:");
							for (const auto& wi : attn_layer->c_proj.get_weights().data)
								file << wi << ", ";
						}
						else if (inner_layer_typeid == typeid(MLP)) {
							auto mlp = dynamic_cast<MLP*>(inner_layer.get());
							file << ".mlp:";
							for (const auto& wi : mlp->mlp.get_weights().data)
								file << wi << ", ";
							file << ("\ntransformer.h.block" + std::to_string(block_idx) + "." + inner_name + ".mlp.bias:");
							for (const auto& wi : mlp->mlp.get_bias().data)
								file << wi << ", ";
							file << ("\ntransformer.h.block" + std::to_string(block_idx) + "." + inner_name + ".mlp_proj:");
							for (const auto& wi : mlp->mlp_proj.get_weights().data)
								file << wi << ", ";
							file << ("\ntransformer.h.block" + std::to_string(block_idx) + "." + inner_name + ".mlp_proj.bias:");
							for (const auto& wi : mlp->mlp_proj.get_bias().data)
								file << wi << ", ";
						}
						file << "\n";
					}
					block_idx++;
				}
			}
			else if (layer_name == "ln_f" && layer_typeid == typeid(LayerNorm)) {
				auto ln_f = dynamic_cast<LayerNorm*>(layer.get());
				file << ("transformer." + layer_name) + ".gamma:";
				for (const auto& wi : ln_f->tmem["gamma"].data)
					file << wi << ", ";
				file << ("\ntransformer." + layer_name) + ".beta:";
				for (const auto& wi : ln_f->tmem["beta"].data)
					file << wi << ", ";
				file << "\n";
			}
		}
		file << "lm_head:";
		for (const auto& wi : lm_head->get_weights().data)
			file << wi << ", ";
		file << "\n";
	}
	file.close();
}


void Model::load_model(std::string model_name) {
	std::ifstream file(model_name + ".csv");

	if (!file.is_open()) {
		std::cerr << "Error: could not open file " << model_name << ".csv because it doesn't exist.\n";
		return;
	}

	std::string line;
	while (std::getline(file, line)) {
		size_t pos = line.find(":");
		if (pos == std::string::npos)
			continue;

		std::string layer_name = line.substr(0, pos) + ':';
		std::string weight_data = line.substr(pos + 1);

		std::vector<dt> data;
		std::stringstream ss(weight_data);
		std::string value;

		while (std::getline(ss, value, ',')) {
			try {
				dt weight = std::stod(value);
				data.push_back(weight);
			}
			catch (...) {
				continue;
			}
		}

		if (data.empty()) {
			std::cerr << "Warning: No data loaded for " << layer_name << "\n";
			continue;
		}

		int vocab_size = this->config.vocab_size;
		int T = this->config.block_size;
		int C = this->config.n_embd;

		if (layer_name == "transformer.wte:") {
			if (data.size() != (size_t)(vocab_size * C)) {
				std::cerr << "Mismatch size for wte: expected " << vocab_size * C << ", got " << data.size() << "\n";
			}
			this->transformer["wte"].get()->weights.data = data;
		}
		else if (layer_name == "transformer.wpe:") {
			if (data.size() != (size_t)(T * C)) {
				std::cerr << "Mismatch size for wpe: expected " << T * C << ", got " << data.size() << "\n";
			}
			this->transformer["wpe"].get()->weights.data = data;
		}
		else if (layer_name.find(".h.") != std::string::npos) {
			int blockIdx = extract_number_between(layer_name, "h.block", ".");
			if (blockIdx == -1) {
				std::cerr << "Block for layer path " << layer_name << " not found\n";
				continue;
			}
			LayerList* layer_list = dynamic_cast<LayerList*>(transformer["h"].get());
			if (!layer_list || blockIdx >= (int)layer_list->blocks.size()) {
				std::cerr << "Invalid block index " << blockIdx << " for " << layer_name << "\n";
				continue;
			}
			Block* block = layer_list->blocks[blockIdx].get();

			std::string block_layer_name = getWordBetweenDelimiters(layer_name, 2, 3);
			auto layer_it = block->layers.find(block_layer_name);
			if (layer_it == block->layers.end()) {
				std::cerr << "Layer " << block_layer_name << " not found in block " << blockIdx << "\n";
				continue;
			}

			Layer* block_layer = layer_it->second.get();

			if (typeid(*block_layer) == typeid(LayerNorm)) {
				LayerNorm* ln = dynamic_cast<LayerNorm*>(block_layer);
				std::string param = getWordBetweenDelimiters(layer_name, 3, 4);
				if (param == "gamma")
					ln->set_gamma(data);
				else if (param == "beta")
					ln->set_beta(data);
			}
			else if (typeid(*block_layer) == typeid(SelfAttention)) {
				SelfAttention* attn_layer = dynamic_cast<SelfAttention*>(block_layer);
				std::string param = getWordBetweenDelimiters(layer_name, 3, 4);
				if (param == "c_attn")
					attn_layer->c_attn.set_weights(data);
				else if (param == "c_proj")
					attn_layer->c_proj.set_weights(data);
			}
			else if (typeid(*block_layer) == typeid(MLP)) {
				MLP* mlp_layer = dynamic_cast<MLP*>(block_layer);
				std::string param_full = getWordBetweenDelimiters(layer_name, 3, 5);
				std::string param_short = getWordBetweenDelimiters(layer_name, 3, 4);

				if (param_full == "mlp.bias")
					mlp_layer->mlp.set_bias(data);
				else if (param_full == "mlp_proj.bias")
					mlp_layer->mlp_proj.set_bias(data);
				else if (param_short == "mlp")
					mlp_layer->mlp.set_weights(data);
				else if (param_short == "mlp_proj")
					mlp_layer->mlp_proj.set_weights(data);
			}
		}
		else if (layer_name.find("ln_f.gamma:") != std::string::npos) {
			dynamic_cast<LayerNorm*>(transformer["ln_f"].get())->set_gamma(data);
		}
		else if (layer_name.find("ln_f.beta:") != std::string::npos) {
			dynamic_cast<LayerNorm*>(transformer["ln_f"].get())->set_beta(data);
		}
		else if (layer_name.find("lm_head:") != std::string::npos) {
			if (data.size() != (size_t)(vocab_size * C)) {
				std::cerr << "Mismatch size for lm_head: expected " << vocab_size * C << ", got " << data.size() << "\n";
			}
			lm_head->set_weights(data);
		}
		else {
			std::cerr << "Unknown layer name: " << layer_name << "\n";
		}
	}

	std::cout << "Model loading complete.\n";
	file.close();
}



