#ifndef __BIN_SAVER_H__
#define __BIN_SAVER_H__

#include<vector>
#include<string>
#include<stdio.h>
#include<map>
#include<functional>

namespace PVX {
	class BinSaver {
		std::vector<long> SizePos;
		FILE* fout;
	public:
		BinSaver(const char* Filename, const char* head);
		BinSaver(const wchar_t* Filename, const char* head);
		BinSaver(const std::string& Filename);
		BinSaver(const char* Filename);
		~BinSaver();
		void Begin(const char* Name);
		void End();
		int Save();
		size_t write(const void* buffer, size_t size, size_t count);
		size_t write(const std::vector<unsigned char>& Bytes);

		template<typename T>
		size_t write(const T& val) { return write(&val, sizeof(T), 1); }

		int OK();

		template<typename T>
		inline size_t Write(const char* Name, const T& v) {
			size_t ret;
			Begin(Name);
			{
				ret = write(&v, sizeof(T), 1);
			} End();
			return ret;
		}

		template<typename T>
		inline size_t Write(const char* Name, const T* Array, size_t Count) {
			size_t ret;
			Begin(Name);
			{
				ret = write(Array, sizeof(T), Count);
			} End();
			return ret;
		}

		template<typename T>
		inline size_t Write(const char* Name, const std::vector<T>& v) {
			size_t ret;
			Begin(Name);
			{
				ret = write(&v[0], sizeof(T), v.size());
			} End();
			return ret;
		}

		inline size_t Write(const char* Name, const std::string& v) {
			size_t ret;
			Begin(Name);
			{
				ret = write(&v[0], 1, v.size());
			} End();
			return ret;
		}
	};

	typedef struct BinHeader {
		union {
			unsigned int iName;
			char sName[4];
		};
		unsigned int Size;
	}BinHeader;

#define TAG(x,y,z,w) (w<<24|z<<16|y<<8|x)

	class BinLoader {
	private:
		FILE* fin;
		size_t cur;
		size_t Size;
		BinLoader* Parent;
		std::map<unsigned int, std::function<void(BinLoader& bin)>> Loader;
		std::function<void(BinLoader& bin, const char*)> AnyLoader = nullptr;
		BinLoader(FILE* fin, size_t Size, BinLoader* Parent);
	public:
		BinLoader(const char* fn, const char* header);
		BinLoader(const wchar_t* fn, const char* header);
		BinLoader(const std::string& fn);
		BinLoader(const char* fn);
		~BinLoader();
		void Process(const char* header, std::function<void(BinLoader&)> Loader);
		void ProcessAny(std::function<void(BinLoader&, const char*)> Loader);
		void Read(void* Data, size_t sz);
		size_t ReadAll(void* Data);
		std::vector<unsigned char> ReadAll();
		std::string RemainingAsString();
		std::wstring RemainingAsWideString();

		template<typename T>
		operator T() { T ret; Read(&ret, sizeof(T)); return ret; }

		void Execute();
		int OK();
		size_t Remaining(int ItemSize = 1);

		void ReadBytes(std::vector<unsigned char>& Bytes, size_t size = 0);

		template<typename T>
		inline void Read(T& val) { Read(&val, sizeof(T)); }

		template<typename T>
		inline void Read(const char* Name, T& val) {
			Process(Name, [&val](BinLoader& b) {
				b.ReadAll(&val);
			});
		}

		template<typename T>
		inline void Read(const char* Name, std::vector<T>& val) {
			Process(Name, [&val](BinLoader& b) {
				size_t sz = b.Remaining(sizeof(T));
				val.resize(sz);
				b.ReadAll(&val[0]);
			});
		}

		inline void Read(const char* Name, std::string& val) {
			Process(Name, [&val](BinLoader& b) {
				size_t sz = b.Remaining();
				val.resize(sz);
				b.ReadAll(&val[0]);
			});
		}
	};
}
#endif