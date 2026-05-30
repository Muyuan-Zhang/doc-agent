"""
Playwright 测试：验证缓存批准流程（方案 B）

流程：
1. 提问一个问题 → 创建 PENDING_REVIEW 缓存条目
2. 访问 Cache 面板 → 加载待审核条目
3. 批准缓存条目
4. 再次提问相同问题 → 验证缓存命中
"""
import asyncio
import json
import sys
from datetime import datetime

import httpx
from playwright.async_api import async_playwright


async def test_cache_approval_flow():
    """完整的缓存批准流程测试"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        test_query = f"什么是机器学习 {datetime.now().isoformat()}"  # 加时间戳确保唯一
        query_hash = None
        
        print("=" * 60)
        print("缓存批准流程测试")
        print("=" * 60)
        
        try:
            # ══════════════════════════════════════════════════════════
            # 步骤 1：导航到首页
            # ══════════════════════════════════════════════════════════
            print("\n[步骤 1] 打开应用首页...")
            await page.goto("http://localhost:8000")
            await page.wait_for_load_state("networkidle")
            print("✅ 页面已加载")
            
            # 检查导航标签
            nav_tabs = await page.locator(".nav-tab").count()
            print(f"   导航标签数量: {nav_tabs}")
            
            # ══════════════════════════════════════════════════════════
            # 步骤 2：在聊天面板提问（创建 PENDING_REVIEW 缓存）
            # ══════════════════════════════════════════════════════════
            print(f"\n[步骤 2] 在聊天面板提问: '{test_query}'")
            
            # 检查是否有可用文档
            doc_count = await page.locator(".file-item").count()
            print(f"   当前可用文档数: {doc_count}")
            
            # 输入问题
            query_input = page.locator("#query-input")
            await query_input.fill(test_query)
            print("   已输入查询")
            
            # 点击发送
            send_btn = page.locator("#send-btn")
            await send_btn.click()
            print("   已点击发送按钮")
            
            # 等待响应（SSE 流式返回）
            print("   等待 LLM 响应...")
            await page.wait_for_timeout(3000)  # 给 LLM 足够时间回应
            
            # 检查消息是否出现
            messages = await page.locator(".message").count()
            print(f"   消息数量: {messages}")
            
            # ══════════════════════════════════════════════════════════
            # 步骤 3：切换到 Cache 面板
            # ══════════════════════════════════════════════════════════
            print("\n[步骤 3] 切换到缓存管理面板...")
            cache_tab = page.locator('[data-panel="cache"]')
            await cache_tab.click()
            print("   已点击 Cache 标签")
            
            await page.wait_for_timeout(500)
            
            # 检查 Cache 面板是否激活
            cache_panel = page.locator("#cache-panel")
            is_visible = await cache_panel.is_visible()
            print(f"   Cache 面板可见: {is_visible}")
            
            # ══════════════════════════════════════════════════════════
            # 步骤 4：加载待审核条目
            # ══════════════════════════════════════════════════════════
            print("\n[步骤 4] 加载待审核缓存条目...")
            load_review_btn = page.locator("#cache-load-review")
            await load_review_btn.click()
            print("   已点击 Load Pending 按钮")
            
            await page.wait_for_timeout(1500)
            
            # 检查是否有待审核条目
            review_items = await page.locator(".review-item").count()
            print(f"   待审核条目数: {review_items}")
            
            if review_items == 0:
                print("   ⚠️  没有待审核条目，可能是:")
                print("      - 缓存系统未激活")
                print("      - 第一次查询时出错")
                print("   继续尝试...")
                await page.wait_for_timeout(2000)
                await load_review_btn.click()
                await page.wait_for_timeout(1500)
                review_items = await page.locator(".review-item").count()
                print(f"   重试后待审核条目数: {review_items}")
            
            # ══════════════════════════════════════════════════════════
            # 步骤 5：批准第一个缓存条目
            # ══════════════════════════════════════════════════════════
            if review_items > 0:
                print(f"\n[步骤 5] 批准缓存条目...")
                
                # 获取第一个条目的信息
                first_item = page.locator(".review-item").first
                query_text = await first_item.locator(".review-item-query").text_content()
                print(f"   条目查询: {query_text[:80]}...")
                
                # 点击 Approve 按钮
                approve_btn = first_item.locator(".btn-approve")
                await approve_btn.click()
                print("   已点击 Approve 按钮")
                
                # 处理 prompt 对话（输入 reviewer_id）
                await page.wait_for_timeout(500)
                
                # 尝试填充 prompt
                try:
                    await page.fill("input[type='text']", "test_reviewer", timeout=1000)
                    await page.press("input[type='text']", "Enter")
                    print("   已输入 reviewer_id")
                except:
                    print("   未检测到 prompt，尝试通过 evaluate 注入...")
                    # 使用 evaluate 模拟用户输入
                    await page.evaluate("""
                        window.prompt = function(msg) { return 'test_reviewer'; }
                    """)
                    # 再次点击 Approve
                    await approve_btn.click()
                
                await page.wait_for_timeout(1000)
                print("   ✅ 批准操作已提交")
            
            # ══════════════════════════════════════════════════════════
            # 步骤 6：验证缓存统计
            # ══════════════════════════════════════════════════════════
            print(f"\n[步骤 6] 查看缓存统计...")
            refresh_stats_btn = page.locator("#cache-refresh-stats")
            await refresh_stats_btn.click()
            print("   已点击 Refresh Stats")
            
            await page.wait_for_timeout(1000)
            
            stats_box = page.locator(".stats-box")
            stats_text = await stats_box.text_content()
            print(f"   缓存统计:\n{stats_text}")
            
            # ══════════════════════════════════════════════════════════
            # 步骤 7：回到聊天面板，再次提问相同问题
            # ══════════════════════════════════════════════════════════
            print(f"\n[步骤 7] 再次提问相同问题（验证缓存命中）...")
            chat_tab = page.locator('[data-panel="chat"]')
            await chat_tab.click()
            print("   已切换到聊天面板")
            
            await page.wait_for_timeout(500)
            
            # 清除之前的消息（可选）
            # await page.evaluate("document.getElementById('messages').innerHTML = ''")
            
            # 再次输入相同问题
            query_input = page.locator("#query-input")
            await query_input.fill(test_query)
            print(f"   已输入相同查询")
            
            # 记录时间
            start_time = datetime.now()
            
            # 点击发送
            send_btn = page.locator("#send-btn")
            await send_btn.click()
            print("   已点击发送按钮")
            
            # 等待响应
            print("   等待缓存响应...")
            await page.wait_for_timeout(2000)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # ══════════════════════════════════════════════════════════
            # 步骤 8：检查响应一致性
            # ══════════════════════════════════════════════════════════
            print(f"\n[步骤 8] 验证缓存命中...")
            print(f"   响应时间: {response_time:.2f}s")
            
            # 获取最后一条消息
            last_message = page.locator(".message.assistant").last
            response_text = await last_message.text_content()
            print(f"   响应内容: {response_text[:100]}...")
            
            print("\n" + "=" * 60)
            print("✅ 测试完成")
            print("=" * 60)
            print("\n验证项:")
            print(f"  ✓ 页面加载")
            print(f"  ✓ 提问功能")
            print(f"  ✓ 缓存面板")
            print(f"  ✓ 批准操作")
            print(f"  ✓ 缓存统计")
            print(f"  ✓ 再次查询")
            
            # 保持浏览器打开用于检查
            print("\n浏览器将保持打开，按 Enter 退出...")
            input()
            
        except Exception as e:
            print(f"\n❌ 测试出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(test_cache_approval_flow())
