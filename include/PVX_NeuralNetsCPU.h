#ifndef __PVX_DEEPNEURALNETS_H__
#define __PVX_DEEPNEURALNETS_H__

#define EIGEN_MPL2_ONLY

#include <Eigen/dense>
#include <vector>
#include <PVX_BinSaver.h>
#include <string>
#include <random>
#include <set>
#include <map>

namespace PVX {
	namespace DeepNeuralNets {
		using netData = Eigen::MatrixXf;

		enum class LayerActivation {
			Tanh,
			TanhBias,
			ReLU,
			Sigmoid,
			Linear
		};

		struct WeightData {
			size_t Offset;
			size_t Count;
			float * Weights;
		};

		class NeuralLayer_Base {
		protected:
			NeuralLayer_Base* PreviousLayer = nullptr;
			std::vector<NeuralLayer_Base*> InputLayers;

			netData output;
			int FeedVersion = -1;
			static int OverrideOnLoad;
			static size_t NextId;
			static float
				__LearnRate,
				__Momentum,
				__iMomentum,
				__RMSprop,
				__iRMSprop,
				__Dropout,
				__iDropout,
				__L2;

			void SetFeedVersion(int ver);

			void Gather(std::set<NeuralLayer_Base*>& g);

			friend class NeuralNetOutput_Base;
			friend class OutputLayer;
			friend class NeuralNetContainer;
			friend class NetContainer;
			
			virtual void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const = 0;

			void FixInputs(const std::vector<NeuralLayer_Base*>& ids);
			std::string name;
			size_t Id = 0;

			int OutputRefCount = 0;
			int RecursionGuard = 0;

			virtual NeuralLayer_Base* newCopy(const std::map<NeuralLayer_Base*,size_t>& IndexOf) = 0;
		public:
			const std::string& Name() const { return name; };
			virtual void DNA(std::map<void*, WeightData> & Weights) = 0;
			void Input(NeuralLayer_Base*);
			void Inputs(const std::vector<NeuralLayer_Base*>&);
			virtual void FeedForward(int) = 0;
			virtual void BackPropagate(const netData &) = 0;
			virtual size_t nInput() const = 0;
			virtual void UpdateWeights() = 0;

			size_t nOutput() const;
			netData Output();
			netData RealOutput();
			size_t BatchSize() const;

			static float LearnRate();
			static void LearnRate(float Alpha);
			static float Momentum();
			static void Momentum(float Beta);
			static float RMSprop();
			static void RMSprop(float Beta);
			static void L2Regularization(float lambda);
			static float L2Regularization();
			static float Dropout();
			static void Dropout(float Rate);
			static void UseDropout(int);
			static void OverrideParamsOnLoad(int ovrd = 1);
			
			virtual void SetLearnRate(float a) = 0;
			virtual void ResetMomentum() = 0;
		};

		netData Concat(const std::vector<netData>& m);

		class InputLayer : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const;
			InputLayer(PVX::BinLoader& bin);
			NeuralLayer_Base* newCopy(const std::map<NeuralLayer_Base*,size_t>& IndexOf);
		public:
			void DNA(std::map<void*, WeightData> & Weights) {};
			InputLayer(const size_t Size);
			InputLayer(const std::string& Name, const size_t Size);
			int Input(const float * Data, int Count = 1);
			int Input(const netData & Data);
			void FeedForward(int) {}
			void BackPropagate(const netData & Gradient) {}
			void UpdateWeights() {};
			size_t nInput() const;

			void InputRaw(const netData & Data);
			netData MakeRawInput(const netData & Data);
			netData MakeRawInput(const float* Data, size_t Count = 1);
			netData MakeRawInput(const std::vector<float>& Input);

			void SetLearnRate(float a) {};
			void ResetMomentum() {};
		};

		enum class TrainScheme {
			Adam,
			RMSprop,
			Momentum,
			AdaGrad,
			Sgd
		};
		class NeuronLayer : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			netData Weights;
			netData DeltaWeights;
			netData RMSprop;
			netData curGradient;
			netData(*Activate)(const netData & Gradient);
			netData(*Derivative)(const netData & Gradient);

			void AdamF(const netData & Gradient);
			void MomentumF(const netData & Gradient);
			void RMSpropF(const netData & Gradient);
			void SgdF(const netData & Gradient);
			void AdaGradF(const netData & Gradient);

			void Adam_WeightDecayF(const netData& Gradient);
			void Momentum_WeightDecayF(const netData& Gradient);
			void RMSprop_WeightDecayF(const netData& Gradient);
			void Sgd_WeightDecayF(const netData& Gradient);
			void AdaGrad_WeightDecayF(const netData& Gradient);

			void(NeuronLayer::*updateWeights)(const netData & Gradient);
			TrainScheme training;
			LayerActivation activation;

			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const;
			static NeuralLayer_Base* Load2(PVX::BinLoader& bin);
			NeuralLayer_Base* newCopy(const std::map<NeuralLayer_Base*,size_t>& IndexOf);
							
			float
				_LearnRate,
				_Momentum,
				_iMomentum,
				_RMSprop,
				_iRMSprop,
				_Dropout,
				_iDropout,
				_L2;
		public:
			NeuronLayer(size_t nInput, size_t nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam);
			NeuronLayer(const std::string& Name, size_t nInput, size_t nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam);
			NeuronLayer(NeuralLayer_Base * inp, size_t nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam);
			NeuronLayer(const std::string& Name, NeuralLayer_Base * inp, size_t nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam);
			size_t nInput() const;

			void FeedForward(int Version);
			void BackPropagate(const netData & TrainData);
			void UpdateWeights();

			void DNA(std::map<void*, WeightData> & Weights);
			void SetLearnRate(float a);
			void ResetMomentum();

			netData & GetWeights();
		};

		class ActivationLayer :NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			netData(*Activate)(const netData& Gradient);
			netData(*Derivative)(const netData& Gradient);
			LayerActivation activation;

			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const;
			static ActivationLayer* Load2(PVX::BinLoader & bin);
			NeuralLayer_Base* newCopy(const std::map<NeuralLayer_Base*,size_t>& IndexOf);
		public:
			ActivationLayer(NeuralLayer_Base* inp, LayerActivation Activation = LayerActivation::ReLU);
			ActivationLayer(size_t inp, LayerActivation Activation = LayerActivation::ReLU);
			ActivationLayer(const std::string& Name, NeuralLayer_Base* inp, LayerActivation Activation = LayerActivation::ReLU);
			ActivationLayer(const std::string& Name, size_t inp, LayerActivation Activation = LayerActivation::ReLU);

			void FeedForward(int Version);
			void BackPropagate(const netData& TrainData);
			void UpdateWeights();
			void DNA(std::map<void*, WeightData>& Weights);
			void SetLearnRate(float a);
			void ResetMomentum();

			size_t nInput() const;
		};

		class NeuronAdder : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const;
			static NeuronAdder* Load2(PVX::BinLoader& bin);
			NeuralLayer_Base* newCopy(const std::map<NeuralLayer_Base*,size_t>& IndexOf);
		public:
			NeuronAdder(const size_t InputSize);
			NeuronAdder(const std::vector<NeuralLayer_Base*> & Inputs);
			NeuronAdder(const std::string& Name, const size_t InputSize);
			NeuronAdder(const std::string& Name, const std::vector<NeuralLayer_Base*>& Inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const netData & Gradient);
			void UpdateWeights();
			size_t nInput() const;

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NeuronMultiplier : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const;
			static NeuronMultiplier* Load2(PVX::BinLoader& bin);
			NeuralLayer_Base* newCopy(const std::map<NeuralLayer_Base*,size_t>& IndexOf);
		public:
			NeuronMultiplier(const size_t inputs);
			NeuronMultiplier(const std::vector<NeuralLayer_Base*> & inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const netData & Gradient);
			void UpdateWeights();
			size_t nInput() const;

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NeuronCombiner : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const;
			static NeuronCombiner* Load2(PVX::BinLoader& bin);
			NeuralLayer_Base* newCopy(const std::map<NeuralLayer_Base*,size_t>& IndexOf);
		public:
			NeuronCombiner(const size_t inputs);
			NeuronCombiner(const std::vector<NeuralLayer_Base*> & inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const netData & Gradient);
			void UpdateWeights();
			size_t nInput() const;

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class RecurrentLayer;

		class RecurrentInput : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			friend class RecurrentLayer;
			int RecurrentNeuronCount;
			netData rnnData;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const;
			NeuralLayer_Base* newCopy(const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			static RecurrentInput* Load2(PVX::BinLoader& bin);
			int BatchSize();
			void FeedIndex(int i);
		public:
			RecurrentInput(NeuralLayer_Base* Input, int RecurrentNeurons);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int);
			void BackPropagate(const netData&);
			size_t nInput() const;
			void UpdateWeights();
			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class RecurrentLayer : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) const;
			NeuralLayer_Base* newCopy(const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			RecurrentInput* RNN_Input;
			static RecurrentLayer* Load2(PVX::BinLoader& bin);
		public:
			RecurrentLayer(NeuralLayer_Base* Input, RecurrentInput* RecurrentInput);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int);
			void BackPropagate(const netData&);
			size_t nInput() const;
			void UpdateWeights();
			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NetDNA {
			std::vector<WeightData> Layers;
			size_t Size = 0;
			friend class NeuralNetOutput_Base; 
			friend class OutputLayer;
		public:
			std::vector<float> GetData();
			void SetData(const float * Data);
		};

		enum class OutputType {
			MeanSquare,
			SoftMax,
			StableSoftMax
		};
		class OutputLayer {
		protected:
			NeuralLayer_Base * LastLayer;
			netData output;
			float Error = -1.0f;
			int Version = 0;
			NetDNA Checkpoint;
			float CheckpointError = -1.0f;
			std::vector<float> CheckpointDNA;
			std::set<NeuralLayer_Base*> Gather();

			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);

			friend class NeuralNetContainer;
			OutputType Type;

			void FeedForwardMeanSquare();
			void FeedForwardSoftMax();
			void FeedForwardStableSoftMax();
			float GetError_MeanSquare(const netData & Data);
			float Train_MeanSquare(const netData & Data);
			float GetError_SoftMax(const netData & Data);
			float Train_SoftMax(const netData & Data);

			std::function<void()> FeedForward;
			std::function<float(const netData&)> GetErrorFnc;
			std::function<float(const netData&)> TrainFnc;
		public:
			OutputLayer(NeuralLayer_Base * Last, OutputType Type = OutputType::MeanSquare);
			float GetError(const netData & Data);
			float Train(const netData & Data);
			float Train(const float * Data);
			void Result(float * Result);
			const netData& Result();
			size_t nOutput();

			void SaveCheckpoint();
			float LoadCheckpoint();

			void ResetMomentum();

			NetDNA GetDNA();
		};

		class ResNetUtility {
			NeuronLayer First, Second;
			NeuronAdder Adder;
			ActivationLayer Activation;
		public:
			ResNetUtility(NeuralLayer_Base* inp, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam);
			ResNetUtility(const ResNetUtility& inp, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam);
			ResNetUtility(const std::string& Name, NeuralLayer_Base* inp, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam);
			ResNetUtility(const std::string& Name, const ResNetUtility& inp, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam);
			NeuralLayer_Base* OutputLayer() const;

			operator NeuralLayer_Base* () { return OutputLayer(); }
		};

		class NetContainer {
		protected:
			std::vector<NeuralLayer_Base*> OwnedLayers;
			std::vector<InputLayer*> Inputs;
			std::vector<NeuronLayer*> DenseLayers;
			NeuralLayer_Base* LastLayer = nullptr;
			OutputType Type;
			netData output;
			mutable int Version = 0;

			std::vector<netData> AllInputData;
			netData AllTrainData;
			std::vector<size_t> TrainOrder;
			int curIteration = 0;
			std::vector<size_t> tmpOrder{ 1, 0 };

			void FeedForwardMeanSquare();
			void FeedForwardSoftMax();
			void FeedForwardStableSoftMax();
			float GetError_MeanSquare(const netData& Data);
			float Train_MeanSquare(const netData& Data);
			float GetError_SoftMax(const netData& Data);
			float Train_SoftMax(const netData& Data);

			std::function<void()> FeedForward;
			std::function<float(const netData&)> GetErrorFnc;
			std::function<float(const netData&)> TrainFnc;

			float error = -1.0f;
		public:
			NetContainer(NeuralLayer_Base* Last, OutputType Type = OutputType::MeanSquare);
			~NetContainer();

			void Save(const std::wstring& Filename);
			void SaveCheckpoint();
			float LoadCheckpoint();

			void ResetMomentum();

			netData MakeRawInput(const netData& inp);
			netData MakeRawInput(const std::vector<float>& inp);
			std::vector<netData> MakeRawInput(const std::vector<netData>& inp);
			netData FromVector(const std::vector<float>& Data);

			std::vector<float> ProcessVec(const std::vector<float>& Inp);

			netData Process(const netData& inp) const;
			netData Process(const std::vector<netData>& inp) const;
			netData ProcessRaw(const netData& inp) const;
			netData ProcessRaw(const std::vector<netData>& inp) const;

			float Train(const netData& inp, const netData& outp);
			float TrainRaw(const netData& inp, const netData& outp);
			float Train(const std::vector<netData>& inp, const netData& outp);
			float TrainRaw(const std::vector<netData>& inp, const netData& outp);

			float Error(const netData& inp, const netData& outp) const;
			float ErrorRaw(const netData& inp, const netData& outp) const;
			float Error(const std::vector<netData>& inp, const netData& outp) const;
			float ErrorRaw(const std::vector<netData>& inp, const netData& outp) const;

			std::vector<std::pair<float*, size_t>> MakeDNA();

			void AddTrainDataRaw(const netData& inp, const netData& outp);
			void AddTrainDataRaw(const std::vector<netData>& inp, const netData& outp);
			void AddTrainData(const netData& inp, const netData& outp);
			void AddTrainData(const std::vector<netData>& inp, const netData& outp);

			void SetBatchSize(int sz);
			float Iterate();
			void CopyWeightsFrom(const NetContainer& from);
		};

		class NeuralNetContainer {
		protected:
			std::vector<NeuralLayer_Base*> OwnedLayers;
			std::vector<InputLayer*> Inputs;
			OutputLayer* Output = nullptr;
			std::vector<netData> AllInputData;
			netData AllTrainData;
			std::vector<size_t> TrainOrder;
			int curIteration = 0;
			std::vector<size_t> tmpOrder{ 1, 0 };
			std::vector<NeuronLayer*> DenseLayers;
		public:
			NeuralNetContainer(OutputLayer* OutLayer);
			NeuralNetContainer(const std::wstring& Filename);
			NeuralNetContainer(const NeuralNetContainer& net);
			~NeuralNetContainer();
			void Save(const std::wstring& Filename);
			void SaveCheckpoint();
			float LoadCheckpoint();
			void ResetMomentum();

			netData MakeRawInput(const netData& inp);
			netData MakeRawInput(const std::vector<float>& inp);
			std::vector<netData> MakeRawInput(const std::vector<netData>& inp);
			netData FromVector(const std::vector<float>& Data);

			std::vector<float> ProcessVec(const std::vector<float>& Inp);

			netData Process(const netData& inp) const;
			netData Process(const std::vector<netData>& inp) const;
			netData ProcessRaw(const netData& inp) const;
			netData ProcessRaw(const std::vector<netData>& inp) const;

			float Train(const netData& inp, const netData& outp);
			float TrainRaw(const netData& inp, const netData& outp);
			float Train(const std::vector<netData>& inp, const netData& outp);
			float TrainRaw(const std::vector<netData>& inp, const netData& outp);

			float Error(const netData& inp, const netData& outp) const;
			float ErrorRaw(const netData& inp, const netData& outp) const;
			float Error(const std::vector<netData>& inp, const netData& outp) const;
			float ErrorRaw(const std::vector<netData>& inp, const netData& outp) const;

			std::vector<std::pair<float*, size_t>> MakeDNA();

			void AddTrainDataRaw(const netData& inp, const netData& outp);
			void AddTrainDataRaw(const std::vector<netData>& inp, const netData& outp);
			void AddTrainData(const netData& inp, const netData& outp);
			void AddTrainData(const std::vector<netData>& inp, const netData& outp);

			void SetBatchSize(int sz);
			float Iterate();
			void CopyWeightsFrom(const NeuralNetContainer& from);
		};
	}
}

#endif