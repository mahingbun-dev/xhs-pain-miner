export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          🔍 XHS Pain Miner
        </h1>
        <p className="text-gray-600 mb-8">
          小红书用户痛点挖掘 — 输入品类关键词，发现用户真实需求
        </p>

        {/* TODO: 品类输入框 */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <input
            type="text"
            placeholder="输入品类关键词，如：防晒霜、婴儿辅食、降噪耳机..."
            className="w-full border border-gray-300 rounded-lg px-4 py-3 text-lg"
          />
        </div>

        {/* TODO: 痛点地图 */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">📊 痛点地图</h2>
          <p className="text-gray-500">输入关键词开始分析...</p>
        </div>

        {/* TODO: 情感趋势 */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">📈 趋势追踪</h2>
          <p className="text-gray-500">分析完成后展示趋势...</p>
        </div>
      </div>
    </main>
  );
}
