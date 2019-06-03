#ifndef __PVX_DEEPNEURALNETS_H__
#define __PVX_DEEPNEURALNETS_H__

#include <Eigen/dense>
#include <vector>
#include <PVX_BinSaver.h>
#include <string>
#include <random>
#include <set>
#include <map>

namespace PVX {
	namespace DeepNeuralNets {
		enum class LayerActivation {
			Tanh,
			TanhBias,
			ReLU,
			Sigmoid,
			Linear
		};

		struct WeightData {
			float * Weights;
			int Offset;
			int Count;
		};

		class NeuralLayer_Base {
		protected:
			NeuralLayer_Base* PreviousLayer = nullptr;
			std::vector<NeuralLayer_Base*> InputLayers;

			Eigen::MatrixXf output;
			int FeedVersion = -1;
			static float
				__LearnRate,
				__Momentum,
				__iMomentum,
				__RMSprop,
				__iRMSprop,
				__Dropout,
				__iDropout;

			void Gather(std::set<NeuralLayer_Base*>& g);

			friend class NeuralNetOutput;
			friend class NeuralNetContainer;
			virtual void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) = 0;

			void FixInputs(const std::vector<NeuralLayer_Base*>& ids);
			std::string name;
		public:
			const std::string& Name() { return name; };
			virtual void DNA(std::map<void*, WeightData> & Weights) = 0;
			void Input(NeuralLayer_Base*);
			void Inputs(const std::vector<NeuralLayer_Base*>&);
			virtual void FeedForward(int) = 0;
			virtual void BackPropagate(const Eigen::MatrixXf &) = 0;
			virtual size_t nInput() = 0;

			virtual void Save(PVX::BinSaver & bin) = 0;
			virtual void Load(PVX::BinLoader & bin) = 0;


			int nOutput();
			int BatchSize();
			Eigen::MatrixXf Output();
			Eigen::MatrixXf RealOutput();

			// Default = 0.0001f
			static float LearnRate();
			static void LearnRate(float Alpha);
			static float Momentum();
			static void Momentum(float Beta);
			static float RMSprop();
			static void RMSprop(float Beta);
			static float Dropout();
			static void Dropout(float Rate);
			static void UseDropout(int);
			
			virtual void SetLearnRate(float a) = 0;
			virtual void ResetMomentum() = 0;
		};

		Eigen::MatrixXf Concat(const std::vector<Eigen::MatrixXf>& m);
		std::vector<float> Divercity(Eigen::MatrixXf& a);
		Eigen::MatrixXf DivercitySort(Eigen::MatrixXf& a, const std::vector<float> & div);


		class InputLayer : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			InputLayer(PVX::BinLoader& bin);
		public:
			void DNA(std::map<void*, WeightData> & Weights) {};
			InputLayer(const size_t Size);
			InputLayer(const std::string& Name, const size_t Size);
			int Input(const float * Data, int Count = 1);
			int Input(const Eigen::MatrixXf & Data);
			void FeedForward(int) {}
			void BackPropagate(const Eigen::MatrixXf & Gradient) {}
			size_t nInput();

			void InputRaw(const Eigen::MatrixXf & Data);
			Eigen::MatrixXf MakeRawInput(const Eigen::MatrixXf & Data);
			Eigen::MatrixXf MakeRawInput(const float* Data, int Count = 1);
			Eigen::MatrixXf MakeRawInput(const std::vector<float>& Input);

			void Save(PVX::BinSaver & bin);
			void Load(PVX::BinLoader & bin);
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
			Eigen::MatrixXf Weights;
			Eigen::MatrixXf DeltaWeights;
			Eigen::MatrixXf RMSprop;
			//void UpdateWeights(const Eigen::MatrixXf & Gradient);
			Eigen::MatrixXf(*Activate)(const Eigen::MatrixXf & Gradient);
			Eigen::MatrixXf(*Derivative)(const Eigen::MatrixXf & Gradient);

			void AdamF(const Eigen::MatrixXf & Gradient);
			void MomentumF(const Eigen::MatrixXf & Gradient);
			//void FirstAdamF(const Eigen::MatrixXf & Gradient);
			//void FirstMomentumF(const Eigen::MatrixXf & Gradient);


			void RMSpropF(const Eigen::MatrixXf & Gradient);
			void SgdF(const Eigen::MatrixXf & Gradient);
			void AdaGradF(const Eigen::MatrixXf & Gradient);

			void(NeuronLayer::*updateWeights)(const Eigen::MatrixXf & Gradient);
			TrainScheme training;
			LayerActivation activation;

			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			static NeuralLayer_Base* Load2(PVX::BinLoader& bin);
							
			float
				_LearnRate,
				_Momentum,
				_iMomentum,
				_RMSprop,
				_iRMSprop,
				_Dropout,
				_iDropout;
		public:
			NeuronLayer(int nInput, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			NeuronLayer(const std::string& Name, int nInput, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			NeuronLayer(NeuralLayer_Base * inp, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			NeuronLayer(const std::string& Name, NeuralLayer_Base * inp, int nOutput, LayerActivation Activate = LayerActivation::ReLU, TrainScheme Train = TrainScheme::Adam, float WeightMax = 1.0f);
			size_t nInput();

			void FeedForward(int Version);
			void BackPropagate(const Eigen::MatrixXf & TrainData);

			void DNA(std::map<void*, WeightData> & Weights);
			void SetLearnRate(float a);
			void ResetMomentum();

			Eigen::MatrixXf & GetWeights();

			void Save(PVX::BinSaver & bin);
			void Load(PVX::BinLoader & bin);
		};

		class ActivationLayer :NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			Eigen::MatrixXf(*Activate)(const Eigen::MatrixXf& Gradient);
			Eigen::MatrixXf(*Derivative)(const Eigen::MatrixXf& Gradient);
			LayerActivation activation;

			void Save(PVX::BinSaver& bin);
			void Load(PVX::BinLoader& bin);

			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			static ActivationLayer* Load2(PVX::BinLoader & bin);
		public:
			ActivationLayer(NeuralLayer_Base* inp, LayerActivation Activation = LayerActivation::ReLU);
			ActivationLayer(int inp, LayerActivation Activation = LayerActivation::ReLU);
			
			void FeedForward(int Version);
			void BackPropagate(const Eigen::MatrixXf& TrainData);
			void DNA(std::map<void*, WeightData>& Weights);
			void SetLearnRate(float a);
			void ResetMomentum();

			// Inherited via NeuralLayer_Base
			size_t nInput();
		};

		class NeuronAdder : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
		public:
			NeuronAdder(const int InputSize);
			NeuronAdder(const std::vector<NeuralLayer_Base*> & Inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const Eigen::MatrixXf & Gradient);
			size_t nInput();

			void Save(PVX::BinSaver & bin);
			void Load(PVX::BinLoader & bin);

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NeuronMultiplier : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
		public:
			NeuronMultiplier(const int inputs);
			NeuronMultiplier(const std::vector<NeuralLayer_Base*> & inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const Eigen::MatrixXf & Gradient);
			size_t nInput();

			void Save(PVX::BinSaver & bin);
			void Load(PVX::BinLoader & bin);

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NeuronCombiner : public NeuralLayer_Base {
		protected:
			friend class NeuralNetContainer;
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
		public:
			NeuronCombiner(const int inputs);
			NeuronCombiner(const std::vector<NeuralLayer_Base*> & inputs);
			void DNA(std::map<void*, WeightData>& Weights);
			void FeedForward(int Version);
			void BackPropagate(const Eigen::MatrixXf & Gradient);
			size_t nInput();

			void Save(PVX::BinSaver & bin);
			void Load(PVX::BinLoader & bin);

			void SetLearnRate(float a);
			void ResetMomentum();
		};

		class NetDNA {
			std::vector<WeightData> Layers;
			int Size;
			friend class NeuralNetOutput;
		public:
			std::vector<float> GetData();
			void SetData(const float * Data);
		};

		class NeuralNetOutput {
		protected:
			NeuralLayer_Base * LastLayer;
			Eigen::MatrixXf output;
			float Error;
			int Version = 0;

			virtual void SaveModel(PVX::BinSaver & bin) = 0;
			virtual void LoadModel(PVX::BinLoader & bin) = 0;
			NetDNA Checkpoint;
			float CheckpointError;
			std::vector<float> CheckpointDNA;
			std::set<NeuralLayer_Base*> Gather();

			virtual void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf) = 0;

			virtual void FeedForward();
			friend class NeuralNetContainer;
		public:
			NeuralNetOutput(NeuralLayer_Base * Last);
			virtual float GetError(const Eigen::MatrixXf & Data) = 0;
			virtual float Train(const float * Data) = 0;
			virtual float Train(const Eigen::MatrixXf & Data) = 0;
			void Result(float * Result);
			const Eigen::MatrixXf& Result();
			int nOutput();

			void Save(const wchar_t * Filename);
			void Load(const wchar_t * Filename);

			void SaveNet(const wchar_t* Filename);
			void LoadNet(const wchar_t* Filename);

			void SaveCheckpoint();
			float LoadCheckpoint();

			void ResetMomentum();

			NetDNA GetDNA();
		};

		class MeanSquareOutput : public NeuralNetOutput {
		protected:
			friend class NeuralNetContainer;
			void SaveModel(PVX::BinSaver & bin);
			void LoadModel(PVX::BinLoader & bin);
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			MeanSquareOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*> & Prevs);
		public:
			MeanSquareOutput(NeuralLayer_Base * Last);
			float GetError(const Eigen::MatrixXf & Data);
			float Train(const float * Data);
			float Train(const Eigen::MatrixXf & Data);
			float Train2(const Eigen::MatrixXf & Data);
		};

		class SoftmaxOutput : public NeuralNetOutput {
		protected:
			friend class NeuralNetContainer;
			void SaveModel(PVX::BinSaver & bin);
			void LoadModel(PVX::BinLoader & bin);
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			void FeedForward();
			SoftmaxOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*>& Prevs);
		public:
			SoftmaxOutput(NeuralLayer_Base * Last);
			float GetError(const Eigen::MatrixXf & Data);
			float Train(const float * Data);
			float Train(const Eigen::MatrixXf & Data);
		};

		class StableSoftmaxOutput : public NeuralNetOutput {
		protected:
			friend class NeuralNetContainer;
			void SaveModel(PVX::BinSaver & bin);
			void LoadModel(PVX::BinLoader & bin);
			void Save(PVX::BinSaver& bin, const std::map<NeuralLayer_Base*, size_t>& IndexOf);
			void FeedForward();
			StableSoftmaxOutput(PVX::BinLoader& bin, const std::vector<NeuralLayer_Base*>& Prevs);
		public:
			StableSoftmaxOutput(NeuralLayer_Base * Last);
			float GetError(const Eigen::MatrixXf & Data);
			float Train(const float * Data);
			float Train(const Eigen::MatrixXf & Data);
		};

		class NeuralNetContainer {
		protected:
			std::vector<NeuralLayer_Base*> Layers;
			std::vector<InputLayer*> Inputs;
			NeuralNetOutput* Output;
			std::vector<std::pair<float*, size_t>> MakeDNA();
			friend class GeneticSolver;
		public:
			NeuralNetContainer(NeuralNetOutput* OutLayer);
			NeuralNetContainer(const std::wstring& Filename);
			~NeuralNetContainer();
			void Save(const std::wstring& Filename);
			void SaveCheckpoint();
			float LoadCheckpoint();
			void ResetMomentum();

			Eigen::MatrixXf MakeRawInput(const Eigen::MatrixXf& inp);
			Eigen::MatrixXf MakeRawInput(const std::vector<float>& inp);
			std::vector<Eigen::MatrixXf> MakeRawInput(const std::vector<Eigen::MatrixXf>& inp);
			Eigen::MatrixXf FromVector(const std::vector<float>& Data);

			std::vector<float> ProcessVec(const std::vector<float>& Inp);

			Eigen::MatrixXf Process(const Eigen::MatrixXf& inp);
			Eigen::MatrixXf Process(const std::vector<Eigen::MatrixXf>& inp);
			Eigen::MatrixXf ProcessRaw(const Eigen::MatrixXf& inp);
			Eigen::MatrixXf ProcessRaw(const std::vector<Eigen::MatrixXf>& inp);

			float Train(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float TrainRaw(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float Train(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);
			float TrainRaw(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);

			float Error(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float ErrorRaw(const Eigen::MatrixXf& inp, const Eigen::MatrixXf& outp);
			float Error(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);
			float ErrorRaw(const std::vector<Eigen::MatrixXf>& inp, const Eigen::MatrixXf& outp);
		};

		class GradientDescent {
			float LearnRate, Momentum, RMSprop, iRMSprop;
			std::function<float()> ErrorFnc;
			std::vector<std::pair<float*, size_t>> Updater;
			std::vector<float> vRMSprop, vMomentum, vGradient, SavePoint;
			size_t ModelSize;
			float Error, LastError;
		public:
			GradientDescent(
				std::function<float()> ErrorFunction,
				std::vector<std::pair<float*, size_t>> Model,
				float LearnRate, float Momentum, float RMSprop
			);
			GradientDescent(
				std::function<float()> ErrorFunction,
				float* Model, size_t ModelSize,
				float LearnRate = 1e-5f, float Momentum = 0.9999f, float RMSprop = 0.9999f
			);
			float Iterate(float dx = 1e-5f);
			void RecalculateError();
			void ClearMomentum();
		};

		class GeneticSolver {
		protected:
			struct ErrorData {
				float Error;
				float * Model;
				int Index;
			};
			size_t ModelSize;

			std::function<float()> ErrorFnc;
			int Population, Survive;
			std::vector<float> Memory;
			std::vector<ErrorData> Generation, Survived;

			std::default_random_engine gen;
			std::uniform_real_distribution<double> dist;
			std::uniform_real_distribution<float> dist01;
			std::uniform_int_distribution<int> intDist;
			int curIter;
			void NextGeneration();
			std::vector<std::pair<float*, size_t>> Updater;

			std::function<float()> NewGenEvent = nullptr;
		public:
			GeneticSolver(
				std::function<float()> ErrorFunction,
				float * Model, size_t ModelSize,
				int Population = 100,
				int Survive = 10,
				float MutationVariance = 1e-5f,
				float MutateProbability = 0.1f,
				float Combine = 0.5f);

			GeneticSolver(
				std::function<float()> ErrorFunction,
				std::vector<std::pair<float*, size_t>> Model,
				int Population = 100,
				int Survive = 10,
				float MutationVariance = 1e-5f,
				float MutateProbability = 0.1f,
				float Combine = 0.5f);

			GeneticSolver(
				std::function<float()> ErrorFunction,
				NeuralNetContainer & Model,
				int Population = 100,
				int Survive = 10,
				float MutationVariance = 1e-5f,
				float MutateProbability = 0.1f,
				float Combine = 0.5f);

			void SetItem(const float * w, int Index = 1, float err = -1.0f);
			float Iterate();
			float Update();
			int BestId();
			float GetItem(int Index);
			void SetItem(int Index);

			void OnNewGeneration(std::function<float()> Event);

			float
				MutationVariance,
				Mutate,
				Combine;

			int GenId;
			float GenPc;
		};
	}
}

#endif