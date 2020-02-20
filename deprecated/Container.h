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