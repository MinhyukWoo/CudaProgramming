#include<cstdio>

struct ExampleStruct
{
	int a;
	float b;
	ExampleStruct() {
		a = 2;
		b = 3.0f;
		c = new char[10] {"hello"};
	}
	void hello() {
		printf("hello\n");
	}
	char * c;
};

void TestStruct();