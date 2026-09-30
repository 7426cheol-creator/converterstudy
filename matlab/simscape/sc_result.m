function res = sc_result(model, builder, expected_source)
%SC_RESULT Empty result record of one Simscape model run.

res = struct();
res.model = model;
res.builder = builder;
res.status = 'NOT_RUN_ENVIRONMENT';
res.reason = '';
res.expected_source = expected_source;
res.result_file = '';
res.env = struct();
res.blocks = struct();
res.solver = '';
res.started_at_local = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
res.runtime_s = NaN;
res.rows = {};
end
