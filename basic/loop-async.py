
import asyncio

async def do_work(name, delay):
    print(f'{name} 시작')
    try:
        await asyncio.sleep(delay)

        if(delay == 2):
            raise Exception('This is the exception you expect to handle')
    except Exception as error:
        print('Caught this error: ' + repr(error))
        return

    print(f'{name} 완료')

async def main():
    # 여러 개의 작업을 동시에 실행합니다.
    tasks = [
        do_work('작업1', 2),
        do_work('작업2', 1),
        do_work('작업3', 3)
    ]
    
    # 모든 작업이 완료될 때까지 기다립니다.
    await asyncio.gather(*tasks)

# 비동기 루프를 실행합니다.
if __name__ == "__main__":
    asyncio.run(main())