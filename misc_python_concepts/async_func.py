# async in python is used to define asynchronous functions
# it allows functions to run concurrently without breaking I/O

# exmaple
import asyncio
import time

async def fetch_data():
    print("Start fetching data...")
    await asyncio.sleep(2)  # Simulate a network delay
    print("Data fetched!")
    return {"data": 123}

async def main():
    print("Main function started.")
    # await is used to call async functions
    # it pauses the execution of the main function until fetch_data is complete
    data = await fetch_data()
    print(f"Received data: {data}")
    print("Main function ended.")
# Running the main function
asyncio.run(main())

# more complex example with multiple async functions
async def task1():
    print("Task 1 started.")
    await asyncio.sleep(1)
    print("Task 1 completed.")
async def task2():
    print("Task 2 started.")
    await asyncio.sleep(1)
    print("Task 2 completed.")
async def task3():
    print("Task 3 started.")
    await asyncio.sleep(1.5)
    print("Task 3 completed.")
async def run_tasks():
    start_time = time.time()
    # gather is used to run multiple async functions concurrently
    # under the hood it creates tasks for each function and runs them in the event loop
    # gather allows multiple functions to be awaited simultaneously
    await asyncio.gather(task1(), task2(), task3())
    end_time = time.time()
    print(f"All tasks completed in {end_time - start_time:.2f} seconds.")
# Running multiple async tasks concurrently
asyncio.run(run_tasks())
# In this example, task1, task2, and task3 run concurrently,

# what if we call run_tast s without asyncio.run
# run_tasks()  # This will not work as expected because run_tasks is an async function

# what if we don;t gather the tasks
async def run_tasks_sequentially():
    start_time = time.time()
    await task1()
    await task2()
    await task3()
    end_time = time.time()
    # This will run the tasks sequentially, one after the other
    print(f"All tasks completed in {end_time - start_time:.2f} seconds.")
asyncio.run(run_tasks_sequentially())


# another example of async function with exception handling
async def faulty_task():
    print("Faulty task started.")
    await asyncio.sleep(1)
    raise ValueError("An error occurred in faulty_task.")

async def run_faulty_task():
    try:
        await faulty_task()
    except ValueError as e:
        print(f"Caught an exception: {e}")

asyncio.run(run_faulty_task())

